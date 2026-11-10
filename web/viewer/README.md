# Data explorer

The explorer is assembled from [geozarr-viewer](https://github.com/lukegre/geozarr-viewer), revision `bdef6756c53c5164c45e9ffbf99e5a0896a0be4a`, using `explorer.toml`. The generated `../explore.html` contains the map and components directly alongside the site navigation. `explorer.js` and `explorer.css` contain the generated runtime and base styles. `page.template.html` defines the project page shell and `../explore.css` provides the shared cream/teal styling. No HighresTA public store is configured yet; the store picker can open a public GeoZarr pyramid in the meantime. A direct link can use `explore.html?store=https%3A%2F%2Fexample.org%2Fdata.zarr` (the viewer also supports `endpoint` for S3 stores).

To configure the eventual default, set `store` in `explorer.toml` to its public URL and rebuild. The configured relative `proxy/` relay serves stores without CORS. Run the bundled local server using `python3 web/viewer/serve.py web --port 8780` and open `http://localhost:8780/explore.html`. A hosted deployment also needs a server providing this relay path.

The viewer requires a **GeoZarr multiscale pyramid** with spatial dimensions, CRS and bounding-box metadata. `src/highres_ta/inference.py` currently writes a plain Zarr v2 store with `talk_median` and `talk_var` (including the `variance_type` dimension), so that output must be converted to a compatible pyramid before publishing. The raw inference store cannot be opened directly by this viewer.

Rebuild with Python 3.11 or newer and a clean checkout of the pinned upstream revision:

```sh
git clone https://github.com/lukegre/geozarr-viewer.git /tmp/geozarr-viewer
git -C /tmp/geozarr-viewer checkout bdef6756c53c5164c45e9ffbf99e5a0896a0be4a
python3 web/viewer/build.py /tmp/geozarr-viewer
python3 web/viewer/serve.py web --port 8780
```

Open `http://localhost:8780/explore.html`. The generated page uses pinned CDN dependencies, so it also needs network access to load the map libraries and basemap. Rebuild and commit `../explore.html`, `explorer.js`, and `explorer.css` whenever the spec changes.
