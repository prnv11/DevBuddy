"""
Serve the docs mirror as two separate local hosts so per-host politeness is
exercised exactly as on the live sites:

    FastAPI -> http://127.0.0.1:8101   (stands in for https://fastapi.tiangolo.com)
    Flask   -> http://127.0.0.1:8102   (stands in for https://flask.palletsprojects.com)

It also reproduces the spider traps of the real sites:
  * old / alternate version trees (/en/2.3.x/..., /en/latest/...) that keep
    generating deeper links,
  * translated copies (/es/, /zh/, /ja/ on FastAPI),
  * search result pages with query strings (?q=...).

Usage: python tools/serve_mirror.py /path/to/mirror
"""
import sys
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

TRAP_PAGE = """<!doctype html><html><head><title>{title}</title></head><body>
<div class="body" role="main"><article class="md-content__inner"><h1>{title}</h1>
<p>This is an alternate copy of the documentation.</p>
<a href="{deeper}">Next section</a> <a href="{path}?highlight=x&amp;page=2">Page 2</a>
</article></div></body></html>"""


class Handler(SimpleHTTPRequestHandler):
    trap_prefixes: tuple = ()

    def log_message(self, *a):
        pass

    def do_GET(self):
        path = urlsplit(self.path).path
        if path.startswith(self.trap_prefixes) or "/search/" in path:
            depth = path.count("/")
            body = TRAP_PAGE.format(title=f"Trap {path}", path=path,
                                    deeper=f"{path.rstrip('/')}/section-{depth}/").encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()


def serve(root: Path, port: int, traps: tuple):
    h = type("H", (Handler,), {"trap_prefixes": traps})
    srv = ThreadingHTTPServer(("127.0.0.1", port), partial(h, directory=str(root)))
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


if __name__ == "__main__":
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "mirror")
    serve(root / "fastapi", 8101, ("/es/", "/zh/", "/ja/"))
    serve(root / "flask", 8102, ("/en/latest/", "/en/3.0.x/", "/en/2.3.x/", "/en/1.1.x/"))
    print("FastAPI mirror on http://127.0.0.1:8101  Flask mirror on http://127.0.0.1:8102  (Ctrl+C to stop)")
    threading.Event().wait()
