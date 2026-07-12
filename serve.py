# -*- coding: utf-8 -*-
"""Dev server: static files from ./webapp + POST /save to dump browser snapshots to disk."""
import base64, http.server, os, socketserver

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "webapp")
SNAP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "snaps")
os.makedirs(SNAP, exist_ok=True)

class H(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=ROOT, **kw)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_POST(self):
        if not self.path.startswith("/save"):
            self.send_error(404); return
        name = self.path.split("=")[-1] if "=" in self.path else "snap"
        name = "".join(c for c in name if c.isalnum() or c in "-_")
        body = self.rfile.read(int(self.headers["Content-Length"])).decode()
        b64 = body.split(",", 1)[1] if "," in body else body
        with open(os.path.join(SNAP, name + ".jpg"), "wb") as f:
            f.write(base64.b64decode(b64))
        self.send_response(200); self.end_headers()
        self.wfile.write(b"ok")

socketserver.TCPServer.allow_reuse_address = True
with socketserver.TCPServer(("127.0.0.1", 8123), H) as srv:
    print("serving on 8123")
    srv.serve_forever()
