#!/usr/bin/env python3
"""Serve an assembled viewer for local use: static files with Range support plus a CORS relay.

    serve.py DIR [--port 8780] [--bind 127.0.0.1]

GET /<file>                  static file from DIR (Range supported; `/` is index.html)
GET /proxy/<https-url>       forwards the GET with its Range header to the https URL and returns the upstream status
                             (200/206/...), Content-Range, Content-Length and body with `Access-Control-Allow-Origin: *`.
                             For stores whose bucket has no CORS rules: the page's spec option `relay = "/proxy/"`.

Only GET (and the CORS OPTIONS preflight) is answered; the viewer never sends HEAD (D-56). Stdlib only.
The relay is for local use: it refuses non-https URLs and hosts that resolve to private/loopback addresses
(use --allow-private-hosts to read a store on your own network).
"""
import argparse
import ipaddress
import mimetypes
import re
import socket
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

CHUNK = 64 * 1024
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/javascript", ".mjs")
mimetypes.add_type("application/json", ".json")
PASS_HEADERS = ("Content-Type", "Content-Length", "Content-Range", "Accept-Ranges", "ETag", "Last-Modified")


def parse_range(header, size):
    """Parse a single `bytes=a-b` / `a-` / `-n` Range header against a file of `size` bytes.

    Returns (start, end) inclusive, None when there is nothing to honour (no header, other unit, several ranges: the
    whole file is served) or "invalid" when it cannot be satisfied (416).
    """
    if not header or not header.startswith("bytes="):
        return None
    spec = header[len("bytes="):].strip()
    if "," in spec:
        return None
    m = re.fullmatch(r"(\d*)-(\d*)", spec)
    if not m or (not m.group(1) and not m.group(2)):
        return "invalid"
    first, last = m.groups()
    if not first:  # suffix: the last n bytes
        n = int(last)
        if n == 0 or size == 0:
            return "invalid"
        return max(0, size - n), size - 1
    start = int(first)
    end = min(int(last), size - 1) if last else size - 1
    if start >= size or end < start:
        return "invalid"
    return start, end


def parse_proxy_target(path, schemes=("https",)):
    """The upstream URL of a `/proxy/<url>` request path (with its query). Raises ValueError when it is not allowed."""
    target = path[len("/proxy/"):] if path.startswith("/proxy/") else ""
    target = re.sub(r"^([a-zA-Z][a-zA-Z0-9+.-]*):/+", r"\1://", target)  # a layer may have collapsed // into /
    u = urlsplit(target)
    if u.scheme not in schemes:
        raise ValueError(f"only {'/'.join(schemes)} URLs can be relayed")
    if not u.hostname:
        raise ValueError("the relayed URL has no host")
    return target


def is_private_host(host):
    """True when `host` is, or resolves to, a loopback/private/link-local/reserved address (or cannot be resolved)."""
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError:
        return True
    for info in infos:
        ip = ipaddress.ip_address(info[4][0].split("%")[0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast or ip.is_unspecified:
            return True
    return False


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):  # hand 3xx back to the page instead of following it past the host check
        return None


class Handler(BaseHTTPRequestHandler):
    directory = "."
    allowed_schemes = ("https",)
    block_private = True
    protocol_version = "HTTP/1.1"
    server_version = "geozarr-viewer-serve"

    # -- plumbing ------------------------------------------------------------------------------------------
    def log_message(self, fmt, *args):
        sys.stderr.write("%s %s\n" % (self.address_string(), fmt % args))

    def cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Expose-Headers", "Content-Range, Content-Length, Accept-Ranges, ETag")

    def reply(self, status, text="", content_type="text/plain; charset=utf-8", extra=()):
        body = text.encode("utf-8")
        self.send_response(status)
        self.cors()
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for k, v in extra:
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    # -- methods -------------------------------------------------------------------------------------------
    def do_GET(self):
        if self.path.startswith("/proxy/"):
            return self.proxy()
        return self.static()

    def do_OPTIONS(self):  # CORS preflight
        self.send_response(204)
        self.cors()
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Range, Content-Type")
        self.send_header("Access-Control-Max-Age", "600")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def method_not_allowed(self):
        self.close_connection = True  # a HEAD answer must not carry a body, so do not reuse the connection for it
        self.reply(405, "GET only (the viewer never sends HEAD)\n", extra=[("Allow", "GET, OPTIONS"), ("Connection", "close")])

    do_HEAD = do_POST = do_PUT = do_DELETE = do_PATCH = method_not_allowed

    # -- static --------------------------------------------------------------------------------------------
    def static(self):
        root = Path(self.directory).resolve()
        rel = unquote(urlsplit(self.path).path).lstrip("/")
        target = (root / rel).resolve()
        if target.is_dir():
            target = target / "index.html"
        try:
            target.relative_to(root)
        except ValueError:
            return self.reply(404, "not found\n")
        if not target.is_file():
            return self.reply(404, "not found\n")
        size = target.stat().st_size
        rng = parse_range(self.headers.get("Range"), size)
        if rng == "invalid":
            return self.reply(416, "range not satisfiable\n", extra=[("Content-Range", f"bytes */{size}")])
        start, end = rng if rng else (0, size - 1)
        ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        self.send_response(206 if rng else 200)
        self.cors()
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Cache-Control", "no-cache")
        if rng:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        length = end - start + 1 if size else 0
        self.send_header("Content-Length", str(length))
        self.end_headers()
        with open(target, "rb") as f:
            f.seek(start)
            left = length
            while left > 0:
                chunk = f.read(min(CHUNK, left))
                if not chunk:
                    break
                self.wfile.write(chunk)
                left -= len(chunk)

    # -- relay ---------------------------------------------------------------------------------------------
    def proxy(self):
        try:
            url = parse_proxy_target(self.path, self.allowed_schemes)
        except ValueError as e:
            return self.reply(400, f"{e}\n")
        host = urlsplit(url).hostname
        if self.block_private and is_private_host(host):
            return self.reply(403, f"refusing to relay to {host} (private or unresolvable address)\n")
        headers = {"User-Agent": "geozarr-viewer-serve", "Accept-Encoding": "identity"}  # identity: lengths stay exact
        if self.headers.get("Range"):
            headers["Range"] = self.headers["Range"]
        req = urllib.request.Request(url, headers=headers, method="GET")
        opener = urllib.request.build_opener(NoRedirect)
        try:
            resp = opener.open(req, timeout=60)
        except urllib.error.HTTPError as e:  # 404, 416, 3xx...: pass the upstream answer on
            resp = e
        except (urllib.error.URLError, OSError, ValueError) as e:
            return self.reply(502, f"upstream request failed: {e}\n")
        try:
            self.send_response(resp.status if hasattr(resp, "status") else resp.code)
            self.cors()
            for name in PASS_HEADERS:
                if resp.headers.get(name):
                    self.send_header(name, resp.headers[name])
            if resp.headers.get("Location"):
                self.send_header("Location", resp.headers["Location"])
            if not resp.headers.get("Content-Length"):
                self.send_header("Connection", "close")  # no length: the body ends with the connection
                self.close_connection = True
            self.end_headers()
            while True:
                chunk = resp.read(CHUNK)
                if not chunk:
                    break
                self.wfile.write(chunk)
        finally:
            resp.close()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Serve a viewer directory with Range support and a /proxy/<https-url> relay.")
    ap.add_argument("directory", help="directory to serve (e.g. the one holding the assembled page)")
    ap.add_argument("--port", type=int, default=8780)
    ap.add_argument("--bind", default="127.0.0.1", help="address to listen on (default: local only)")
    ap.add_argument("--allow-private-hosts", action="store_true", help="let /proxy/ reach private/loopback addresses")
    args = ap.parse_args(argv)
    d = Path(args.directory)
    if not d.is_dir():
        print(f"error: not a directory: {d}", file=sys.stderr)
        return 2
    handler = type("ViewerHandler", (Handler,), {"directory": str(d), "block_private": not args.allow_private_hosts})
    srv = ThreadingHTTPServer((args.bind, args.port), handler)
    print(f"serving {d.resolve()} on http://{args.bind}:{args.port}/  (relay: /proxy/<https-url>)", file=sys.stderr)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
