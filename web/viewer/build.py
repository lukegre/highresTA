#!/usr/bin/env python3
"""Assemble GeoZarr components directly into the HighresTA explorer page."""
import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REVISION = "bdef6756c53c5164c45e9ffbf99e5a0896a0be4a"
HERE = Path(__file__).resolve().parent


def integrate(source):
    css = re.search(r"<style>(.*?)</style>", source, re.S).group(1)
    body = re.search(r"<body>(.*?)</body>", source, re.S).group(1)
    module = re.search(r'<script type="module">(.*?)</script>', body, re.S)
    viewer_js = module.group(1)
    # The upstream viewer prioritizes the URL and saved preference; set the
    # project default only when neither has been supplied.
    original = "app.storage.get(`centerlon:${cfg.title}`) ?? 0"
    if original not in viewer_js:
        raise ValueError("The viewer centre-longitude initialization has changed")
    viewer_js = viewer_js.replace(original, "app.storage.get(`centerlon:${cfg.title}`) ?? 205", 1)
    (HERE / "explorer.js").write_text(viewer_js)
    body = body[:module.start()] + body[module.end():]
    # A containing section owns the map layout in the project page.
    body = body.replace('<main class="gv-main">', '<div class="gv-main">').replace("</main>", "</div>")
    (HERE / "explorer.css").write_text(css)
    template = (HERE / "page.template.html").read_text()
    (HERE.parent / "explore.html").write_text(template.replace("<!-- GEOZARR_VIEWER -->", body.strip()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("viewer_repo", type=Path, help="Pinned geozarr-viewer checkout")
    args = parser.parse_args()
    repo = args.viewer_repo.resolve()
    revision = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        parser.error(f"Expected geozarr-viewer {REVISION}, got {revision}")
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "viewer.html"
        subprocess.run([sys.executable, str(repo / "scripts/assemble.py"), str(HERE / "explorer.toml"), "-o", str(output)], check=True)
        integrate(output.read_text())


if __name__ == "__main__":
    main()
