"""Reproduce saved-run data preparation and orchestrate model validation reports."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from .dataio import load_data
from .quantile_ensemble import QuantileRegressionEnsemble
from .training import data_fingerprint, prepare_data, resolve_path
from .validation import evaluate_model, swap_salinity
from .validation_report import write_validation_reports


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_validation(
    model_path: Path, output_dir: Path, config: dict, settings: dict | None = None,
) -> tuple[Path, Path]:
    """Write predictions, scores, provenance, PDF and HTML for one trained run.

    A run's ``configs.json`` defines data preparation; current validation settings
    control reporting. This prevents later training configuration edits from silently
    changing the observations addressed by the saved positional split row IDs.
    """
    model_path, output_dir = Path(model_path), Path(output_dir)
    settings = dict(settings or config.get("validation", {}))
    aggregation = settings.get("aggregation", "median")
    if aggregation not in {"mean", "median"}:
        raise ValueError("Validation aggregation must be 'mean' or 'median'.")
    max_cpus = int(settings.get("max_cpus", 42))
    if max_cpus < 1:
        raise ValueError("max_cpus must be positive.")
    coastal_depth = float(settings.get("coastal_depth", 500))
    if not 0 < coastal_depth < float("inf"):
        raise ValueError("coastal_depth must be positive and finite.")
    run_dir = model_path.parent.parent
    snapshot = run_dir / "configs.json"
    training_config = json.loads(snapshot.read_text()) if snapshot.exists() else config
    if not snapshot.exists():
        logger.warning("No saved configs.json; using supplied data configuration.")
    data = prepare_data(training_config)
    fingerprint = data_fingerprint(data)
    manifest_path = model_path.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    expected = manifest.get("data_fingerprint")
    if expected and fingerprint != expected:
        raise ValueError("Prepared data fingerprint differs from the saved training run; refusing positional validation.")
    if not expected:
        logger.warning("No manifest data fingerprint; only split row/cruise alignment can be checked.")

    model = QuantileRegressionEnsemble.load(model_path)
    model.optimise_cpus(max_cpus)
    split_files = list((run_dir / "splits").glob("split_*.parquet"))
    products = list(dict.fromkeys(["salinity", *settings.get("salinity_products", ["salinity", "salt_soda", "sss_multiobs", "sss_cci"])]))
    variants = {"salinity": data}
    if any(product != "salinity" for product in products):
        raw = load_data(str(resolve_path(training_config["paths"]["training_data"])))
        cfg = training_config["data"]
        for product in products:
            if product != "salinity":
                variants[product] = swap_salinity(
                    data, raw, product, cfg["coordinate_columns"],
                    cfg["minimum_salinity"], cfg["maximum_salinity"],
                )
    logger.info("Validating {} members across {} salinity products", len(model.estimators), len(variants))
    predictions, scores = evaluate_model(
        model, data, training_config, split_files, variants,
        aggregation=aggregation, depth_threshold=coastal_depth,
    )
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_path": str(model_path.resolve()),
        "model_sha256": _sha256(model_path),
        "training_config": str(snapshot.resolve()) if snapshot.exists() else "supplied config",
        "training_data": str(resolve_path(training_config["paths"]["training_data"])),
        "data_fingerprint": fingerprint,
        "fingerprint_verified": bool(expected),
        "n_observations": len(data),
        "n_members": len(model.estimators),
        "n_splits": len(split_files),
        "aggregation": aggregation,
        "saved_aggregation": model.aggregation,
        "coastal_depth": coastal_depth,
        "salinity_products": list(variants),
        "residual_definition": "predicted minus observed (µmol kg⁻¹)",
        "test_method": "Common held-out test rows, all ensemble members.",
        "validation_method": "Out-of-fold validation uses only members trained for the corresponding split.",
        "alternate_salinity_method": "Original rows retained; replacement outside training salinity bounds or missing is excluded per product. Product sample counts can differ; comparisons are not paired.",
        "split_files": [str(path.resolve()) for path in sorted(split_files)],
        "validation_settings": settings,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(output_dir / "predictions.parquet", index=False)
    scores.to_csv(output_dir / "scores.csv", index=False)
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    result = write_validation_reports(predictions, scores, output_dir, metadata)
    logger.success("Validation reports saved to {}", output_dir)
    return result
