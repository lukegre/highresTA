"""Validate a saved ensemble and write PDF and standalone interactive HTML reports."""

from pathlib import Path
from typing import Annotated

import typer

from highres_ta.training import CONFIG_DIR, load_config, resolve_path
from highres_ta.validation_runner import run_validation

app = typer.Typer()


@app.command(help=__doc__)
def main(
    config_path: Annotated[Path, typer.Option("--config", exists=True)] = CONFIG_DIR,
    overrides: Annotated[
        list[Path],
        typer.Option(
            "--override",
            help="YAML file deep-merged over the configuration; repeatable, applied in order.",
            exists=True,
            dir_okay=False,
            readable=True,
        ),
    ] = [],
    model_path: Annotated[Path | None, typer.Option(help="Defaults to <paths.output_dir>/models/quantile_ensemble.joblib.")] = None,
    output_dir: Annotated[Path | None, typer.Option(help="Defaults to <model run>/validation.")] = None,
    max_cpus: Annotated[int | None, typer.Option(min=1)] = None,
    aggregation: Annotated[str | None, typer.Option(help="Member aggregation: median or mean.")] = None,
) -> None:
    config = load_config(config_path, overrides)
    settings = dict(config.get("validation", {}))
    if max_cpus is not None:
        settings["max_cpus"] = max_cpus
    if aggregation is not None:
        settings["aggregation"] = aggregation
    model_path = resolve_path(model_path or Path(config["paths"]["output_dir"]) / "models" / "quantile_ensemble.joblib")
    destination = resolve_path(output_dir) if output_dir else model_path.parent.parent / "validation"
    pdf, html = run_validation(model_path, destination, config, settings)
    typer.echo(f"PDF: {pdf}\nHTML: {html}")


if __name__ == "__main__":
    app()
