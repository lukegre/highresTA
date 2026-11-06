import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import optuna
import pandas as pd

from highres_ta import QuantileRegressionEnsemble, decompose_variance
from highres_ta.train_test_split import (
    SPLIT_NAMES,
    outer_blocks,
    split_indices,
    tuning_folds,
    validate_split,
)
from highres_ta.training import (
    PARAMETER_BOUNDS,
    Member,
    catboost_params,
    decode_dropped_features,
    dropped_feature_options,
    members,
    optimize,
    rashomon_table,
    select_rashomon_members,
    selected_params,
    suggest_parameters,
    trial_distance,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)

QUANTILES = [0.1, 0.5, 0.9]


FEATURES = [
    "salinity", "temperature", "nitrate", "ssh_adt", "chl", "mld", "sla", "wind", "par",
    "sst_anom", "ncoord_x", "ncoord_y",
]


def make_config(n_parts=7, n_members_per_split=7, epsilon=0.02, max_dropped=2):
    return {
        "cross_validation": {"n_parts": n_parts, "split_random_state": 42},
        "data": {"feature_names": FEATURES, "target_name": "talk"},
        "model": {
            "linear_features": ["salinity"],
            "polynomial_degree": 1,
            "quantiles": QUANTILES,
            "maximum_iterations": 100,
            "early_stopping_rounds": 10,
        },
        "optimization": {
            "protected_features": ["salinity", "temperature"],
            "max_dropped_features": max_dropped,
            "n_trials": 6,
            "n_startup_trials": 3,
            "parallel_trials": 1,
            "threads_per_trial": 2,
        },
        "ensemble": {
            "aggregation": "mean",
            "n_members_per_split": n_members_per_split,
            "rashomon_epsilon": epsilon,
        },
    }


def make_data(n_cruises=70, rows_per_cruise=8, seed=0):
    rng = np.random.default_rng(seed)
    expocodes = np.repeat([f"CRUISE{i:03d}" for i in range(n_cruises)], rows_per_cruise)
    cruise_salinity = rng.uniform(30.0, 37.0, n_cruises)
    salinity = np.repeat(cruise_salinity, rows_per_cruise) + rng.normal(
        0, 0.1, n_cruises * rows_per_cruise
    )
    index = pd.MultiIndex.from_arrays(
        [expocodes, np.arange(len(expocodes))], names=["expocode", "row"]
    )
    return pd.DataFrame({"salinity": salinity, "talk": 2300 + 50 * salinity}, index=index)


class FixedQuantileEstimator:
    def __init__(self, prediction, quantiles):
        self.prediction = np.asarray(prediction, dtype=float)
        self.quantiles_ = np.asarray(quantiles, dtype=float)

    def predict_quantiles(self, X):
        return self.prediction[: len(X)]


class MemberTests(unittest.TestCase):
    def test_full_design_is_six_splits_by_seven_members(self):
        selected = members(make_config())

        self.assertEqual(len(selected), 42)
        self.assertEqual(
            [(m.split, m.replicate) for m in selected],
            [(k, r) for k in range(1, 7) for r in range(7)],
        )
        self.assertEqual(len({m.seed for m in selected}), 42)
        self.assertEqual(len({m.name for m in selected}), 42)

    def test_member_seed_and_name(self):
        member = Member(split=3, replicate=5)

        self.assertEqual(member.seed, 305)
        self.assertEqual(member.name, "split_3_member_5")

    def test_split_filter(self):
        by_split = members(make_config(), split=2)

        self.assertEqual([m.replicate for m in by_split], list(range(7)))
        self.assertTrue(all(m.split == 2 for m in by_split))

    def test_out_of_range_split_raises(self):
        for split in (0, 7):
            with self.subTest(split=split), self.assertRaises(ValueError):
                members(make_config(), split=split)


class DroppedFeatureTests(unittest.TestCase):
    def test_option_count_and_contents(self):
        options = dropped_feature_options(make_config())

        self.assertEqual(len(options), 1 + 10 + 45)
        self.assertEqual(len(set(options)), len(options))
        self.assertIn("none", options)
        for option in options:
            names = decode_dropped_features(option)
            self.assertNotIn("salinity", names)
            self.assertNotIn("temperature", names)

    def test_round_trip_and_feature_order(self):
        for option in dropped_feature_options(make_config()):
            names = decode_dropped_features(option)
            self.assertEqual("+".join(names) or "none", option)
            positions = [FEATURES.index(name) for name in names]
            self.assertEqual(positions, sorted(positions))
        self.assertEqual(decode_dropped_features("none"), [])
        self.assertEqual(decode_dropped_features("nitrate+ssh_adt"), ["nitrate", "ssh_adt"])

    def test_max_dropped_one(self):
        self.assertEqual(len(dropped_feature_options(make_config(max_dropped=1))), 11)


def make_trial(
    number, value, dropped="none", state=optuna.trial.TrialState.COMPLETE, user_attrs=None, **overrides
):
    params = {
        "learning_rate": 0.1,
        "l2_leaf_reg": 10.0,
        "depth": 6,
        "random_strength": 5.0,
    } | overrides
    options = dropped_feature_options(make_config())
    distributions = {
        name: (
            optuna.distributions.IntDistribution(int(low), int(high))
            if isinstance(low, int)
            else optuna.distributions.FloatDistribution(low, high, log=log)
        )
        for name, (low, high, log) in PARAMETER_BOUNDS.items()
    }
    distributions["dropped_features"] = optuna.distributions.CategoricalDistribution(options)
    params["dropped_features"] = dropped
    trial = optuna.trial.create_trial(
        params=params,
        distributions=distributions,
        value=value if state == optuna.trial.TrialState.COMPLETE else None,
        state=state,
        user_attrs=user_attrs,
    )
    trial.number = number
    return trial


class RashomonSelectionTests(unittest.TestCase):
    def numbers(self, selection):
        return [trial.number for trial, _ in selection]

    def test_epsilon_filters_and_best_is_first(self):
        trials = [
            make_trial(0, 1.05, learning_rate=0.2),
            make_trial(1, 1.00),
            make_trial(2, 1.01, learning_rate=0.05),
            make_trial(3, 1.02, learning_rate=0.03),
        ]
        selection = select_rashomon_members(trials, make_config(n_members_per_split=3))

        self.assertEqual(selection[0][0].number, 1)
        self.assertNotIn(0, self.numbers(selection[:3]))
        self.assertTrue(all(flag for _, flag in selection))
        self.assertEqual(sorted(self.numbers(selection)), [1, 2, 3])

    def test_greedy_prefers_distant_over_close(self):
        trials = [
            make_trial(0, 1.000),
            make_trial(1, 1.001, learning_rate=0.11),
            make_trial(2, 1.010, dropped="nitrate+ssh_adt", learning_rate=0.03, depth=4),
        ]
        selection = select_rashomon_members(trials, make_config(n_members_per_split=2))

        self.assertEqual(self.numbers(selection), [0, 2])

    def test_ties_break_by_loss(self):
        trials = [make_trial(0, 1.000), make_trial(1, 1.010), make_trial(2, 1.005)]
        selection = select_rashomon_members(trials, make_config(n_members_per_split=2))

        self.assertEqual(self.numbers(selection), [0, 2])

    def test_pruned_and_failed_trials_are_excluded(self):
        trials = [
            make_trial(0, 1.0),
            make_trial(1, None, state=optuna.trial.TrialState.PRUNED, learning_rate=0.03),
            make_trial(2, None, state=optuna.trial.TrialState.FAIL, learning_rate=0.2),
            make_trial(3, 1.01, learning_rate=0.05),
        ]
        selection = select_rashomon_members(trials, make_config(n_members_per_split=2))

        self.assertEqual(sorted(self.numbers(selection)), [0, 3])

    def test_fills_outside_set_when_too_small(self):
        trials = [
            make_trial(0, 1.0),
            make_trial(1, 1.5, learning_rate=0.05),
            make_trial(2, 1.2, learning_rate=0.2),
            make_trial(3, 2.0, learning_rate=0.03),
        ]
        selection = select_rashomon_members(trials, make_config(n_members_per_split=3))

        self.assertEqual(self.numbers(selection), [0, 2, 1])
        self.assertEqual([flag for _, flag in selection], [True, False, False])

    def test_too_few_completed_trials_raises(self):
        trials = [make_trial(0, 1.0), make_trial(1, None, state=optuna.trial.TrialState.PRUNED)]
        with self.assertRaises(ValueError):
            select_rashomon_members(trials, make_config(n_members_per_split=2))


class TrialDistanceTests(unittest.TestCase):
    def test_identical_trials_have_zero_distance(self):
        self.assertEqual(trial_distance(make_trial(0, 1.0), make_trial(1, 1.0)), 0.0)

    def test_opposite_extremes_have_unit_distance(self):
        low = {name: low for name, (low, _, _) in PARAMETER_BOUNDS.items()}
        high = {name: high for name, (_, high, _) in PARAMETER_BOUNDS.items()}
        a = make_trial(0, 1.0, dropped="nitrate+ssh_adt", **low)
        b = make_trial(1, 1.0, dropped="chl+mld", **high)

        self.assertAlmostEqual(trial_distance(a, b), 1.0)

    def test_symmetric_and_bounded(self):
        a = make_trial(0, 1.0, dropped="nitrate", learning_rate=0.05, depth=4)
        b = make_trial(1, 1.0, dropped="nitrate+chl", l2_leaf_reg=50.0)

        self.assertAlmostEqual(trial_distance(a, b), trial_distance(b, a))
        self.assertTrue(0.0 <= trial_distance(a, b) <= 1.0)


class RashomonOutputTests(unittest.TestCase):
    def test_table_columns_for_fill_in_scenario(self):
        config = make_config(n_members_per_split=3)
        trials = [
            make_trial(0, 1.0),
            make_trial(1, 1.5, learning_rate=0.05),
            make_trial(2, 1.2, learning_rate=0.2),
            make_trial(3, 2.0, learning_rate=0.03),
        ]
        table = rashomon_table(trials, select_rashomon_members(trials, config), config)

        self.assertEqual(table["trial"].tolist(), [0, 2, 1])  # trial 3 is neither in set nor selected
        self.assertEqual(table["rank"].tolist(), [1, 2, 3])
        self.assertEqual(table["member"].tolist(), [0, 1, 2])
        self.assertEqual(table["in_rashomon_set"].tolist(), [True, False, False])
        self.assertEqual(table["selected"].tolist(), [True, True, True])

    def test_tie_break_is_deterministic(self):
        config = make_config(n_members_per_split=2)
        for order in ([5, 2, 9], [9, 2, 5], [2, 9, 5]):
            trials = [make_trial(n, 1.0) for n in order]
            selection = select_rashomon_members(trials, config)
            self.assertEqual([t.number for t, _ in selection], [2, 5])

    def test_negative_epsilon_raises(self):
        with self.assertRaises(ValueError):
            select_rashomon_members([make_trial(0, 1.0)], make_config(n_members_per_split=1, epsilon=-0.1))

    def test_selected_params(self):
        config = make_config()
        trial = make_trial(0, 1.0, dropped="nitrate+ssh_adt", user_attrs={"refit_iterations": 37})
        params = selected_params(trial, config, seed=305)

        self.assertEqual(params["ignored_features"], ["nitrate", "ssh_adt"])
        self.assertNotIn("dropped_features", params)
        self.assertNotIn("early_stopping_rounds", params)
        self.assertEqual(params["iterations"], 37)
        self.assertEqual(params["random_seed"], 305)
        self.assertEqual(params["thread_count"], 1 * 2)

        none = selected_params(make_trial(1, 1.0, user_attrs={"refit_iterations": 5}), config, seed=1)
        self.assertNotIn("ignored_features", none)
        self.assertNotIn("ignored_features", catboost_params(make_trial(1, 1.0).params))


class OptimizeTests(unittest.TestCase):
    def fake_objective_factory(self, data, folds, config, seed):
        def objective(trial):
            params = suggest_parameters(trial, config)
            trial.report(params["depth"], step=0)
            if trial.number % 4 == 3:
                raise optuna.TrialPruned()
            trial.set_user_attr("refit_iterations", 10)
            return params["learning_rate"] + params["depth"]

        return objective

    def test_resume_and_signature(self):
        config = make_config()
        data = make_data(n_cruises=4, rows_per_cruise=2)
        folds = [(np.arange(4), np.arange(4, 8)), (np.arange(4, 8), np.arange(4))]
        with tempfile.TemporaryDirectory() as tmp, mock.patch(
            "highres_ta.training.objective_factory", self.fake_objective_factory
        ):
            path = Path(tmp) / "studies" / "split_1.sqlite3"
            study = optimize(data, folds, config, 1, path)
            complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            self.assertGreaterEqual(len(complete), 6)
            self.assertTrue(any(t.state == optuna.trial.TrialState.PRUNED for t in study.trials))

            again = optimize(data, folds, config, 1, path)
            self.assertEqual(len(again.trials), len(study.trials))

            with self.assertRaises(ValueError):
                optimize(data, folds[:1], config, 1, path)


class PartitionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = make_config()
        cls.data = make_data()
        cls.blocks = outer_blocks(cls.data, cls.config)
        cls.groups = cls.data.index.get_level_values("expocode").to_numpy()

    def test_outer_blocks_partition_rows_by_cruise(self):
        self.assertEqual(len(self.blocks), 7)
        covered = np.concatenate(self.blocks)
        self.assertEqual(len(covered), len(self.data))
        self.assertEqual(len(np.unique(covered)), len(self.data))
        cruise_sets = [set(self.groups[block]) for block in self.blocks]
        for i in range(len(cruise_sets)):
            for j in range(i + 1, len(cruise_sets)):
                self.assertFalse(cruise_sets[i] & cruise_sets[j])

    def test_outer_blocks_are_deterministic(self):
        again = outer_blocks(self.data, self.config)
        for left, right in zip(self.blocks, again):
            np.testing.assert_array_equal(np.sort(left), np.sort(right))

    def test_outer_blocks_require_three_parts(self):
        with self.assertRaises(ValueError):
            outer_blocks(self.data, make_config(n_parts=2))

    def test_test_block_is_shared_and_validation_rotates(self):
        for k in range(1, 7):
            with self.subTest(split=k):
                split = split_indices(self.blocks, k)
                np.testing.assert_array_equal(split["test"], np.sort(self.blocks[0]))
                np.testing.assert_array_equal(split["validation"], np.sort(self.blocks[k]))
                self.assertEqual(len(split["train_blocks"]), 5)
                np.testing.assert_array_equal(
                    split["train"], np.sort(np.concatenate(split["train_blocks"]))
                )

    def test_split_is_disjoint_complete_and_leak_free(self):
        for k in range(1, 7):
            with self.subTest(split=k):
                split = split_indices(self.blocks, k)
                covered = np.concatenate([split[name] for name in SPLIT_NAMES])
                self.assertEqual(len(covered), len(self.data))
                self.assertEqual(len(np.unique(covered)), len(self.data))
                for name in SPLIT_NAMES:
                    self.assertTrue(np.all(np.diff(split[name]) > 0))
                validate_split(self.data, split, k)

    def test_invalid_split_raises(self):
        for k in (0, 7):
            with self.subTest(split=k), self.assertRaises(ValueError):
                split_indices(self.blocks, k)

    def test_tuning_folds_are_leave_one_block_out_within_train(self):
        for k in range(1, 7):
            with self.subTest(split=k):
                split = split_indices(self.blocks, k)
                train_rows = split["train"]
                folds = tuning_folds(split)

                self.assertEqual(len(folds), 5)
                for (fit, holdout), block in zip(folds, split["train_blocks"]):
                    np.testing.assert_array_equal(train_rows[holdout], np.sort(block))
                    self.assertEqual(len(fit) + len(holdout), len(train_rows))
                    self.assertFalse(set(fit) & set(holdout))
                    self.assertFalse(
                        set(self.groups[train_rows[fit]]) & set(self.groups[train_rows[holdout]])
                    )
                holdouts = np.concatenate([holdout for _, holdout in folds])
                np.testing.assert_array_equal(np.sort(holdouts), np.arange(len(train_rows)))

    def test_tuning_folds_are_reproducible(self):
        split = split_indices(self.blocks, 3)
        first = tuning_folds(split)
        second = tuning_folds(split_indices(self.blocks, 3))
        for (fit_a, holdout_a), (fit_b, holdout_b) in zip(first, second):
            np.testing.assert_array_equal(fit_a, fit_b)
            np.testing.assert_array_equal(holdout_a, holdout_b)


class DecompositionTests(unittest.TestCase):
    def test_terms_sum_to_total_member_variance(self):
        rng = np.random.default_rng(1)
        predictions = rng.normal(size=(6, 7, 11, 3))

        result = decompose_variance(predictions)
        flat = predictions.reshape(42, 11, 3)

        np.testing.assert_allclose(result["mean"], flat.mean(axis=0))
        np.testing.assert_allclose(result["split_var"] + result["hpo_var"], flat.var(axis=0))
        self.assertTrue(np.all(result["split_var"] >= 0))
        self.assertTrue(np.all(result["hpo_var"] >= 0))

    def test_constant_within_split_has_no_hpo_variance(self):
        split_offsets = np.arange(6, dtype=float)[:, None, None, None]
        predictions = np.broadcast_to(split_offsets, (6, 7, 4, 3)).copy()

        result = decompose_variance(predictions)

        np.testing.assert_allclose(result["hpo_var"], 0.0)
        np.testing.assert_allclose(result["split_var"], np.var(np.arange(6.0)))

    def test_identical_splits_have_no_split_variance(self):
        rng = np.random.default_rng(2)
        one_split = rng.normal(size=(1, 7, 4, 3))
        predictions = np.repeat(one_split, 6, axis=0)

        result = decompose_variance(predictions)

        np.testing.assert_allclose(result["split_var"], 0.0, atol=1e-12)
        np.testing.assert_allclose(result["hpo_var"], one_split[0].var(axis=0))

    def test_rejects_wrong_rank(self):
        with self.assertRaises(ValueError):
            decompose_variance(np.zeros((42, 4, 3)))


class EnsembleDecompositionTests(unittest.TestCase):
    n_splits = 3
    n_replicates = 4
    n_rows = 5

    def make_ensemble(self, predictions, order=None):
        design = [(k, r) for k in range(1, self.n_splits + 1) for r in range(self.n_replicates)]
        if order is not None:
            design = [design[i] for i in order]
        estimators = [
            FixedQuantileEstimator(predictions[k - 1, r], QUANTILES) for k, r in design
        ]
        return QuantileRegressionEnsemble(
            estimators=estimators,
            seeds=[100 * k + r for k, r in design],
            quantiles=QUANTILES,
            member_splits=[k for k, _ in design],
            member_replicates=[r for _, r in design],
        )

    def make_predictions(self, seed=3):
        rng = np.random.default_rng(seed)
        base = np.sort(rng.normal(size=(self.n_splits, self.n_replicates, self.n_rows, 3)), axis=-1)
        return base

    def test_predict_decomposition_matches_function(self):
        predictions = self.make_predictions()
        ensemble = self.make_ensemble(predictions)
        X = np.zeros((self.n_rows, 1))

        result = ensemble.predict_decomposition(X)
        expected = decompose_variance(predictions)

        for key in ("mean", "split_var", "hpo_var"):
            np.testing.assert_allclose(result[key], expected[key], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(
            result["mean"], ensemble.predict_quantiles(X), rtol=1e-5, atol=1e-6
        )

    def test_predict_decomposition_is_independent_of_member_order(self):
        predictions = self.make_predictions()
        order = np.random.default_rng(4).permutation(self.n_splits * self.n_replicates)
        ensemble = self.make_ensemble(predictions, order=list(order))

        result = ensemble.predict_decomposition(np.zeros((self.n_rows, 1)))
        expected = decompose_variance(predictions)

        for key in ("mean", "split_var", "hpo_var"):
            np.testing.assert_allclose(result[key], expected[key], rtol=1e-5, atol=1e-6)

    def test_predict_decomposition_requires_member_design(self):
        prediction = np.zeros((self.n_rows, 3))
        ensemble = QuantileRegressionEnsemble(
            estimators=[FixedQuantileEstimator(prediction, QUANTILES) for _ in range(2)],
            seeds=[0, 1],
            quantiles=QUANTILES,
        )
        with self.assertRaises(ValueError):
            ensemble.predict_decomposition(np.zeros((self.n_rows, 1)))

    def test_predict_decomposition_rejects_unbalanced_design(self):
        prediction = np.zeros((self.n_rows, 3))
        ensemble = QuantileRegressionEnsemble(
            estimators=[FixedQuantileEstimator(prediction, QUANTILES) for _ in range(3)],
            seeds=[100, 101, 200],
            quantiles=QUANTILES,
            member_splits=[1, 1, 2],
            member_replicates=[0, 1, 0],
        )
        with self.assertRaises(ValueError):
            ensemble.predict_decomposition(np.zeros((self.n_rows, 1)))

    def test_rejects_mismatched_design_lengths(self):
        prediction = np.zeros((self.n_rows, 3))
        estimators = [FixedQuantileEstimator(prediction, QUANTILES) for _ in range(2)]
        bad_designs = (
            {"member_splits": [1], "member_replicates": [0, 1]},
            {"member_splits": [1, 1], "member_replicates": [0]},
            {"member_splits": [1, 1]},
            {"member_replicates": [0, 1]},
        )
        for kwargs in bad_designs:
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                QuantileRegressionEnsemble(
                    estimators=estimators, seeds=[100, 101], quantiles=QUANTILES, **kwargs
                )


if __name__ == "__main__":
    unittest.main()
