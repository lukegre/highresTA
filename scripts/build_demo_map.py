#!/usr/bin/env python3
"""Render a dated HighresTA Zarr field for the website's map card."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / 'web'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--store', required=True, help='Path or URL to a HighresTA Zarr store')
    p.add_argument('--date', required=True, help='Requested date (YYYY-MM-DD)')
    p.add_argument('--group', default=None, help='Optional Zarr group, e.g. pyramid level')
    p.add_argument('--nearest', action='store_true', help='Use the nearest available time instead of requiring an exact date')
    p.add_argument('--output-dir', type=Path, default=WEB / 'maps', help='Destination for PNG and WebP')
    p.add_argument('--vmin', type=float, default=1900)
    p.add_argument('--vmax', type=float, default=2500)
    p.add_argument('--update-site', action='store_true', help='Update overview.maps[1] in web/content.json')
    args = p.parse_args()
    if args.vmin >= args.vmax:
        p.error('--vmin must be less than --vmax')
    try:
        date = pd.Timestamp(args.date)
    except ValueError:
        p.error('--date must be a valid date')
    if date.tz is not None or date != date.normalize():
        p.error('--date must be a date without a time or timezone')
    out = args.output_dir.expanduser().resolve()
    if args.update_site and not out.is_relative_to(WEB.resolve()):
        p.error('--update-site requires --output-dir inside the web directory')
    ds = xr.open_zarr(args.store, group=args.group)
    if 'talk_median' not in ds:
        p.error(f'talk_median not found; available: {list(ds.data_vars)}')
    da = ds['talk_median']
    if not {'time', 'lat', 'lon'}.issubset(da.dims):
        p.error('variable must have time, lat and lon dimensions')
    extra = set(da.dims) - {'time', 'lat', 'lon'}
    if extra:
        p.error(f'variable has unsupported dimensions: {sorted(extra)}')
    try:
        da = da.sel(time=date, **({'method': 'nearest'} if args.nearest else {})).transpose('lat', 'lon').compute()
    except KeyError:
        p.error('date not found in store; use --nearest to select the nearest available time')
    actual = pd.Timestamp(da.time.item())
    if not np.isfinite(da.values).any():
        p.error('selected field has no finite values')
    if not (np.isfinite(da.lat.values).all() and np.isfinite(da.lon.values).all()):
        p.error('latitude and longitude coordinates must be finite')
    # Normalize longitude for a conventional geographic view, then sort coordinates.
    da = da.assign_coords(lon=((da.lon + 180) % 360) - 180).sortby('lon').sortby('lat')
    stamp = actual.strftime('%Y-%m-%d')
    display = f'{actual.day} {actual.strftime("%B %Y")}'
    out.mkdir(parents=True, exist_ok=True)
    full = out / f'alkalinity-{stamp}.png'
    thumb = out / f'alkalinity-{stamp}.webp'
    coast = json.loads((ROOT / 'src/highres_ta/assets/ne_110m_coastline.json').read_text())
    x = np.array([v if v is not None else np.nan for v in coast['lon']])
    y = np.array([v if v is not None else np.nan for v in coast['lat']])
    x[np.r_[False, np.abs(np.diff(x)) > 180]] = np.nan
    fig, ax = plt.subplots(figsize=(14, 7), layout='constrained')
    fig.patch.set_facecolor('#f8fafb')
    ax.set_facecolor('#e3e8e8')
    im = ax.pcolormesh(da.lon, da.lat, da, cmap='viridis', vmin=args.vmin, vmax=args.vmax, shading='nearest', rasterized=True)
    ax.plot(x, y, color='#536971', lw=.32)
    ax.set(xlim=(-180, 180), ylim=(-90, 90), xlabel='Longitude', ylabel='Latitude')
    ax.set_xticks([-180, -120, -60, 0, 60, 120, 180])
    ax.set_yticks([-60, -30, 0, 30, 60])
    ax.grid(color='white', alpha=.22, lw=.5)
    ax.set_title(f'Estimated surface total alkalinity | {display}', loc='left', fontsize=17, fontweight='bold', color='#173b49', pad=14)
    bar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=.11, shrink=.68, aspect=38, extend='both')
    bar.set_label('Total alkalinity (µmol kg⁻¹)')
    fig.savefig(full, dpi=160, facecolor=fig.get_facecolor())
    # Capture only the plotted map for the landing card, without title or colorbar.
    fig.canvas.draw()
    fig.savefig(thumb.with_suffix('.thumbnail.png'), dpi=100, bbox_inches=ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted()), pad_inches=0)
    plt.close(fig)
    tmp = thumb.with_suffix('.thumbnail.png')
    try:
        with Image.open(tmp) as image:
            image.save(thumb, 'WEBP', quality=88, method=6)
    finally:
        tmp.unlink(missing_ok=True)
    if args.update_site:
        content_path = WEB / 'content.json'
        content = json.loads(content_path.read_text())
        card = content['overview']['maps'][1]
        card.update(image=Path(os.path.relpath(thumb, WEB.resolve())).as_posix(), full=Path(os.path.relpath(full, WEB.resolve())).as_posix(),
                    alt=f'Estimated surface total alkalinity on {display}.',
                    legend=f'{args.vmin:,.0f} to {args.vmax:,.0f} µmol kg⁻¹',
                    description=f'Surface alkalinity on {display}, from the stored talk_median model field.')
        content_path.write_text(json.dumps(content, ensure_ascii=False, indent=2) + '\n')
    print(f'Selected {actual.isoformat()}\nFull map: {full}\nThumbnail: {thumb}')
    if args.update_site:
        print(f'Updated {WEB / "content.json"}')


if __name__ == '__main__':
    main()
