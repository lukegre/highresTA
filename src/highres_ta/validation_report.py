"""Generate PDF and offline interactive reports for model validation."""

from __future__ import annotations

import json
import textwrap
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages


_REQUIRED_PREDICTIONS = {
    "dataset", "salinity_product", "observed", "predicted", "residual",
    "lon", "lat", "bottomdepth", "split_id", "row_id",
}
_METRICS = [
    "n", "root_mean_squared_error", "mean_absolute_error", "median_absolute_error",
    "r2_score", "mean_bias", "median_bias", "huber_loss",
]
_LABELS = {
    "dataset": "Dataset", "salinity_product": "Salinity product", "region": "Region",
    "n": "N", "root_mean_squared_error": "RMSE", "mean_absolute_error": "MAE",
    "median_absolute_error": "Median AE", "r2_score": "R²", "mean_bias": "Mean bias",
    "median_bias": "Median bias", "huber_loss": "Huber loss",
}
_PROVENANCE_ORDER = [
    "created_utc", "model_path", "model_sha256", "training_config", "training_data",
    "data_fingerprint", "fingerprint_verified", "n_observations", "n_members", "n_splits",
    "aggregation", "saved_aggregation", "coastal_depth", "salinity_products",
    "residual_definition", "test_method", "validation_method", "alternate_salinity_method",
    "validation_settings", "split_files",
]


@lru_cache(maxsize=1)
def _coastlines() -> dict:
    """Bundled public-domain Natural Earth coastlines keep HTML fully offline."""
    return json.loads(files("highres_ta").joinpath("assets/ne_110m_coastline.json").read_text())


def _map_axes(ax: Any) -> None:
    coast = _coastlines()
    ax.plot(np.asarray(coast["lon"], dtype=float), np.asarray(coast["lat"], dtype=float),
            color="#444444", linewidth=.5, zorder=3)
    ax.set(xlim=(-180, 180), ylim=(-90, 90), xlabel="Longitude (°E)", ylabel="Latitude (°N)")
    ax.set_aspect("equal", adjustable="box")


def _validate(predictions: pd.DataFrame, scores: pd.DataFrame) -> None:
    missing = _REQUIRED_PREDICTIONS - set(predictions.columns)
    if missing:
        raise ValueError(f"predictions missing required columns: {', '.join(sorted(missing))}")
    missing = {"dataset", "salinity_product", "region"} - set(scores.columns)
    if missing:
        raise ValueError(f"scores missing required columns: {', '.join(sorted(missing))}")


def _fmt(value: Any) -> str:
    if pd.isna(value):
        return "—"
    if isinstance(value, (float, np.floating)):
        return f"{value:.3f}"
    return str(value)


def _meta_text(value: Any) -> str:
    if isinstance(value, dict):
        return "; ".join(f"{key}={item}" for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return ", ".join(map(str, value))
    return str(value)


def _score_table(ax: Any, frame: pd.DataFrame, title: str) -> None:
    cols = [c for c in ["region", *_METRICS] if c in frame]
    ax.axis("off")
    ax.set_title(title, loc="left", fontweight="bold", fontsize=12, pad=12)
    if frame.empty:
        ax.text(.02, .5, "No scores available", transform=ax.transAxes)
        return
    values = [[_fmt(v) for v in row] for row in frame[cols].itertuples(index=False, name=None)]
    table = ax.table(cellText=values, colLabels=[_LABELS.get(c, c) for c in cols],
                     colWidths=[.2 if c == "region" else .1 for c in cols],
                     loc="upper left", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)


def _write_pdf(pred: pd.DataFrame, scores: pd.DataFrame, path: Path, metadata: dict[str, str]) -> None:
    with PdfPages(path) as pdf:
        # Provenance uses wrapped label/value lines so long split-file lists remain readable.
        lines = ["Provenance and methods", ""]
        keys = [key for key in _PROVENANCE_ORDER if key in metadata]
        keys.extend(key for key in metadata if key not in keys)
        for key in keys:
            value = metadata[key]
            if key == "split_files" and isinstance(value, list):
                value = f"{len(value)} split files: " + ", ".join(Path(x).name for x in value)
            label = key.replace("_", " ").capitalize()
            lines.extend(textwrap.wrap(f"{label}: {_meta_text(value)}", width=104, subsequent_indent="    ") or [""])
        lines.extend([
            "", f"Prediction rows: {len(pred):,}",
            "Datasets: " + ", ".join(map(str, sorted(pred.dataset.dropna().unique()))),
            "Salinity products: " + ", ".join(map(str, sorted(pred.salinity_product.dropna().unique()))),
            "Residual convention: predicted − observed.",
            "All finite scored observations are included in the charts and HTML report.",
        ])
        for start in range(0, len(lines), 29):
            fig = plt.figure(figsize=(11.7, 8.3))
            title = "Total Alkalinity Model Validation" if start == 0 else "Validation provenance (continued)"
            fig.suptitle(title, fontsize=20, y=.96)
            ax = fig.add_axes([.08, .08, .84, .80]); ax.axis("off")
            ax.text(0, 1, "\n".join(lines[start:start + 29]), va="top", fontsize=10,
                    linespacing=1.35, wrap=True)
            pdf.savefig(fig); plt.close(fig)

        # One score page per data/product pair avoids shrinking long tables to fit.
        score_groups = list(scores.groupby(["dataset", "salinity_product"], dropna=False, sort=True))
        if not score_groups:
            score_groups = [(('', ''), scores)]
        for (dataset, product), frame in score_groups:
            fig, ax = plt.subplots(figsize=(11.7, 8.3))
            _score_table(ax, frame, f"Regional scores · {dataset} · {product}")
            pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)

        for (dataset, product), group in pred.groupby(["dataset", "salinity_product"], dropna=False, sort=True):
            finite = group.replace([np.inf, -np.inf], np.nan).dropna(subset=["observed", "predicted", "residual"])
            fig, axes = plt.subplots(2, 2, figsize=(11.7, 8.3))
            fig.suptitle(f"{dataset} · {product} (n={len(finite):,})", fontsize=15)
            if finite.empty:
                for axis in axes.flat:
                    axis.axis("off")
                axes[0, 0].text(.5, .5, "No finite predictions", ha="center")
            else:
                ax = axes[0, 0]
                ax.scatter(finite.observed, finite.predicted, s=7, alpha=.35, rasterized=True)
                lo = float(min(finite.observed.min(), finite.predicted.min()))
                hi = float(max(finite.observed.max(), finite.predicted.max()))
                ax.plot([lo, hi], [lo, hi], "k--", lw=1)
                ax.set(xlabel="Observed TA (µmol/kg)", ylabel="Predicted TA (µmol/kg)", title="Predicted vs observed")
                ax.set_aspect("equal", adjustable="box")
                axes[0, 1].hist(finite.residual, bins=45, color="#4878a8", alpha=.85)
                axes[0, 1].axvline(0, color="black", ls="--", lw=1)
                axes[0, 1].set(title="Residual distribution", xlabel="Predicted − observed (µmol/kg)", ylabel="Count")
                sc = axes[1, 0].scatter(finite.lon, finite.lat, c=finite.residual, s=8,
                                        cmap="coolwarm", vmin=-30, vmax=30, alpha=.8, rasterized=True)
                axes[1, 0].set(xlim=(-180, 180), ylim=(-90, 90), xlabel="Longitude", ylabel="Latitude",
                               title="Residuals by location")
                _map_axes(axes[1, 0])
                fig.colorbar(sc, ax=axes[1, 0], label="Predicted − observed")
                axes[1, 1].scatter(finite.predicted, finite.residual, s=7, alpha=.35, rasterized=True)
                axes[1, 1].axhline(0, color="black", ls="--", lw=1)
                axes[1, 1].set(xlabel="Predicted TA (µmol/kg)", ylabel="Residual (µmol/kg)", title="Residual vs predicted")
            fig.tight_layout(rect=(0, 0, 1, .94)); pdf.savefig(fig); plt.close(fig)

        # Compare products on the global/all row when that row exists.
        if "root_mean_squared_error" in scores and scores.salinity_product.nunique(dropna=False) > 1:
            global_rows = scores[scores.region.astype(str).str.lower().isin(["global", "all"])]
            if not global_rows.empty:
                pivot = global_rows.pivot_table(index="salinity_product", columns="dataset",
                                                values="root_mean_squared_error", aggfunc="first")
                fig, ax = plt.subplots(figsize=(9, 5)); pivot.plot.bar(ax=ax)
                ax.set(title="Alternate salinity product comparison · global RMSE", ylabel="RMSE", xlabel="Salinity product")
                fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        if not pred.empty:
            baseline = str(metadata.get("baseline_salinity_product", "salinity"))
            test = pred[(pred.dataset == "test") & (pred.salinity_product == baseline)].drop_duplicates("row_id")
            if not test.empty:
                depth = float(metadata.get("coastal_depth", 500))
                coastal = pd.to_numeric(test.bottomdepth, errors="coerce").abs() <= depth
                highlat = pd.to_numeric(test.lat, errors="coerce").abs() >= 60
                category = np.select([coastal & highlat, coastal, highlat],
                                     ["Coastal + high latitude", "Coastal", "High latitude"], default="Other")
                fig, ax = plt.subplots(figsize=(11.7, 6.2))
                for name in ["Other", "Coastal", "High latitude", "Coastal + high latitude"]:
                    keep = category == name
                    if keep.any():
                        ax.scatter(test.loc[keep, "lon"], test.loc[keep, "lat"], s=7, alpha=.7,
                                   label=f"{name} (n={int(keep.sum()):,})", rasterized=True)
                ax.set(xlim=(-180, 180), ylim=(-90, 90), xlabel="Longitude", ylabel="Latitude",
                       title=f"Baseline test locations and region masks (coastal depth ≤ {depth:g} m)")
                _map_axes(ax)
                ax.legend(loc="upper center", bbox_to_anchor=(.5, -.12), ncol=2, frameon=False)
                fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

        if {"reference_salinity", "salinity_delta"}.issubset(pred.columns):
            for product, group in pred.groupby("salinity_product", dropna=False, sort=True):
                if str(product) == "salinity":
                    continue
                mapped = group.drop_duplicates("row_id").replace([np.inf, -np.inf], np.nan)
                mapped = mapped.dropna(subset=["lon", "lat", "salinity_delta"])
                if mapped.empty:
                    continue
                fig, ax = plt.subplots(figsize=(11.7, 6.2))
                sc = ax.scatter(mapped.lon, mapped.lat, c=mapped.salinity_delta, s=8,
                                cmap="coolwarm", vmin=-.4, vmax=.4, rasterized=True)
                ax.set(xlim=(-180, 180), ylim=(-90, 90), xlabel="Longitude", ylabel="Latitude",
                       title=f"{product} minus reference salinity (unique rows, n={len(mapped):,})")
                _map_axes(ax)
                fig.colorbar(sc, ax=ax, label="Salinity difference (psu)")
                fig.tight_layout(); pdf.savefig(fig); plt.close(fig)

def _html_report(pred: pd.DataFrame, scores: pd.DataFrame, metadata: dict[str, str]) -> str:
    # pandas' JSON encoder maps NaN/NaT to JSON null, including nullable split IDs.
    predictions_json = json.loads(pred.to_json(orient="records", date_format="iso"))
    scores_json = json.loads(scores.to_json(orient="records", date_format="iso"))
    payload = json.dumps({"predictions": predictions_json, "scores": scores_json, "coastlines": _coastlines(),
                          "metadata": metadata}, ensure_ascii=False).replace("</", "<\\/")
    from plotly.offline import get_plotlyjs

    return _HTML.replace("__PLOTLY__", get_plotlyjs()).replace("__DATA__", payload)


_HTML = r'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Validation report</title><style>
body{font:14px system-ui,sans-serif;color:#17212b;margin:24px auto;padding:0 20px;max-width:1320px;background:#f6f8fa}h1{margin-bottom:4px}section{background:white;padding:18px;margin:14px 0;border:1px solid #dce2e7;border-radius:8px}label{margin-right:14px}select{padding:6px}#chart{width:100%;max-width:1200px;aspect-ratio:4/3;margin:0 auto}table{border-collapse:collapse;width:100%;display:block;overflow:auto;max-height:520px}th,td{border:1px solid #d8dee4;padding:5px 8px;text-align:right;white-space:nowrap}th{position:sticky;top:0;background:#edf2f6}td:first-child,th:first-child{text-align:left}.muted{color:#576574}</style></head><body>
<h1>Total Alkalinity Model Validation</h1><details><summary>Provenance and methods</summary><div id="provenance" class="muted"></div></details><div id="report-status" class="muted">Loading report…</div>
<section><h2>Explore predictions</h2><label>Dataset <select id="ds"></select></label><label>Salinity product <select id="sp"></select></label><label>Map color <select id="mapcolor"><option value="residual">Residual</option><option value="salinity_delta">Salinity difference</option></select></label><p class="muted">Hover for observation details. Plotly toolbar supports zoom, pan, box/lasso select, and image export.</p><div id="chart"></div></section>
<section><h2>Regional scores</h2><div id="scoretable"></div></section>
<script>__PLOTLY__</script><script>const DATA=__DATA__,P=DATA.predictions,S=DATA.scores;
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const uniq=(a,k)=>[...new Set(a.map(x=>x[k]).filter(x=>x!==null&&x!==undefined))].sort((a,b)=>String(a).localeCompare(String(b)));
function fill(id,items){document.getElementById(id).innerHTML=items.map(x=>`<option>${esc(x)}</option>`).join('')}
fill('ds',[...new Set([...uniq(P,'dataset'),...uniq(S,'dataset')])].sort());
fill('sp',[...new Set([...uniq(P,'salinity_product'),...uniq(S,'salinity_product')])].sort());
document.getElementById('provenance').innerHTML=Object.entries(DATA.metadata||{}).map(([k,v])=>`<b>${esc(k.replaceAll('_',' '))}:</b> ${esc(v)}`).join(' · ');
function table(rows,cols){return '<table><thead><tr>'+cols.map(c=>`<th>${esc(c)}</th>`).join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+cols.map(c=>`<td>${esc(r[c])}</td>`).join('')+'</tr>').join('')+'</tbody></table>'}
const layout={autosize:true,showlegend:false,margin:{l:65,r:60,t:90,b:55},hovermode:'closest',
  xaxis:{title:{text:'Observed TA (µmol/kg)'},domain:[0,.46],anchor:'y'},yaxis:{title:{text:'Predicted TA (µmol/kg)'},domain:[.56,1],anchor:'x'},
  xaxis2:{title:{text:'Predicted − observed (µmol/kg)'},domain:[.54,1],anchor:'y2'},yaxis2:{title:{text:'Count'},domain:[.56,1],anchor:'x2'},
  xaxis3:{title:{text:'Longitude (°E)'},range:[-180,180],constrain:'domain',domain:[0,.46],anchor:'y3'},yaxis3:{title:{text:'Latitude (°N)'},range:[-90,90],scaleanchor:'x3',scaleratio:1,constrain:'domain',domain:[0,.44],anchor:'x3'},
  xaxis4:{title:{text:'Predicted TA (µmol/kg)'},domain:[.54,1],anchor:'y4'},yaxis4:{title:{text:'Residual (µmol/kg)'},domain:[0,.44],anchor:'x4'},
  annotations:[{text:'Observed vs predicted',xref:'paper',yref:'paper',x:.23,y:1.08,showarrow:false},{text:'Residual distribution',xref:'paper',yref:'paper',x:.77,y:1.08,showarrow:false},{text:'Residual map',xref:'paper',yref:'paper',x:.23,y:.47,showarrow:false},{text:'Residual vs predicted',xref:'paper',yref:'paper',x:.77,y:.47,showarrow:false}]};
function render(){const ds=document.getElementById('ds').value,sp=document.getElementById('sp').value;
 const rows=P.filter(r=>String(r.dataset)===ds&&String(r.salinity_product)===sp);
 const finite=rows.filter(r=>[r.observed,r.predicted,r.residual,r.lon,r.lat].every(Number.isFinite));
 const custom=finite.map(r=>[r.row_id,r.split_id,r.observed,r.predicted,r.residual,r.lon,r.lat,r.bottomdepth,r.salinity_delta]);
 const hover='row %{customdata[0]}<br>split %{customdata[1]}<br>observed %{customdata[2]}<br>predicted %{customdata[3]}<br>residual %{customdata[4]}<br>salinity difference %{customdata[8]}<br>lon %{customdata[5]}, lat %{customdata[6]}<br>bottom depth %{customdata[7]}<extra></extra>';
 const traces=[
  {type:'scattergl',mode:'markers',name:'Predicted vs observed',x:finite.map(r=>r.observed),y:finite.map(r=>r.predicted),customdata:custom,hovertemplate:hover,marker:{size:5,opacity:.5,color:'#276b8e'},xaxis:'x',yaxis:'y'},
  {type:'histogram',name:'Residual distribution',x:finite.map(r=>r.residual),nbinsx:45,marker:{color:'#4878a8'},xaxis:'x2',yaxis:'y2'},
  {type:'scattergl',mode:'markers',name:'Residual map',x:finite.map(r=>r.lon),y:finite.map(r=>r.lat),customdata:custom,hovertemplate:hover,marker:{size:6,opacity:.8,color:finite.map(r=>r[document.getElementById('mapcolor').value]??r.residual),colorscale:[[0,'#2166ac'],[.5,'#f7f7f7'],[1,'#b2182b']],cmin:document.getElementById('mapcolor').value==='salinity_delta' ? -.4 : -30,cmax:document.getElementById('mapcolor').value==='salinity_delta' ? .4 : 30,colorbar:{title:{text:document.getElementById('mapcolor').value==='salinity_delta'?'Salinity difference':'Residual'}}},xaxis:'x3',yaxis:'y3'},
  {type:'scattergl',mode:'markers',name:'Residual vs predicted',x:finite.map(r=>r.predicted),y:finite.map(r=>r.residual),customdata:custom,hovertemplate:hover,marker:{size:5,opacity:.5,color:'#276b8e'},xaxis:'x4',yaxis:'y4'},
  {type:'scatter',mode:'lines',name:'Coastlines',x:DATA.coastlines.lon,y:DATA.coastlines.lat,line:{color:'#444444',width:.8},hoverinfo:'skip',xaxis:'x3',yaxis:'y3'}];
 const obsMin=finite.length?Math.min(...finite.map(r=>r.observed)):0,obsMax=finite.length?Math.max(...finite.map(r=>r.observed)):1;
 const predMin=finite.length?Math.min(...finite.map(r=>r.predicted)):0,predMax=finite.length?Math.max(...finite.map(r=>r.predicted)):1;
 Plotly.react('chart',traces,{...layout,title:{text:`${esc(ds)} · ${esc(sp)} (n=${rows.length.toLocaleString()})`},
  shapes:[{type:'line',xref:'x',yref:'y',x0:obsMin,x1:obsMax,y0:obsMin,y1:obsMax,line:{dash:'dash',color:'#444'}},
  {type:'line',xref:'x4',yref:'y4',x0:predMin,x1:predMax,y0:0,y1:0,line:{dash:'dash',color:'#444'}}]},{responsive:true,scrollZoom:true,displaylogo:false})
  .then(()=>document.getElementById('report-status').textContent='Report ready')
  .catch(err=>document.getElementById('report-status').textContent='Chart error: '+err.message);
 const selected=S.filter(r=>String(r.dataset)===ds&&String(r.salinity_product)===sp);
 document.getElementById('scoretable').innerHTML=table(selected,Object.keys(S[0]||{}));
}
document.getElementById('ds').onchange=render;document.getElementById('sp').onchange=render;document.getElementById('mapcolor').onchange=render;
if(!P.some(r=>Number.isFinite(r.salinity_delta)))document.getElementById('mapcolor').disabled=true;render();</script></body></html>'''


def write_validation_reports(
    predictions: pd.DataFrame,
    scores: pd.DataFrame,
    output_dir: Path,
    metadata: dict,
) -> tuple[Path, Path]:
    """Write a multipage PDF and standalone interactive HTML report."""
    _validate(predictions, scores)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "validation.pdf"
    html_path = output_dir / "validation.html"
    safe_metadata = {str(key): value for key, value in (metadata or {}).items()}
    _write_pdf(predictions, scores, pdf_path, safe_metadata)
    html_path.write_text(_html_report(predictions, scores, safe_metadata), encoding="utf-8")
    return pdf_path, html_path
