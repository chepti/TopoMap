# -*- coding: utf-8 -*-
"""Local studio server.

Serves the viewer from ./webapp on http://localhost:8123 and adds a local-only
admin studio at /admin for creating new sites: upload map images, mark
elevation anchors, build (pipeline/run_all.py) and publish to Hostinger.

Local endpoints (not available on the public site):
  POST /save?name=x            debug snapshot -> snaps/x.jpg
  POST /upload?site=x          multipart images -> input/x/
  GET  /sites-input            list input sites + their files
  POST /classify?site=x        auto-classify roles of uploaded images
  POST /build                  json body {site,name,config} -> run pipeline (async)
  GET  /build-log              current build log + status
  POST /publish?site=x         scp data/<site> + sites.json + index.html to Hostinger
  GET  /input/<site>/<file>    serve uploaded images
"""
import base64, json, os, re, shutil, subprocess, threading
import http.server, socketserver

ROOT = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(ROOT, "webapp")
SNAP = os.path.join(ROOT, "snaps")
INPUT = os.path.join(ROOT, "input")
os.makedirs(SNAP, exist_ok=True)
os.makedirs(INPUT, exist_ok=True)

SSH_CONFIG = r"T:\.ssh\config"
REMOTE = "hostinger"
REMOTE_DIR = "~/public_html/maps"

job = {"running": False, "log": "", "ok": None, "site": None}

def safe_id(s):
    return re.sub(r"[^a-zA-Z0-9_-]", "", s or "")[:40]

def run_job(kind, args_list, site):
    def worker():
        job.update(running=True, log="", ok=None, site=site)
        try:
            p = subprocess.Popen(args_list, cwd=ROOT, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True,
                                 encoding="utf-8", errors="replace")
            for line in p.stdout:
                job["log"] += line
            p.wait()
            job["ok"] = (p.returncode == 0)
            job["log"] += f"\n== {kind} {'הסתיים בהצלחה' if job['ok'] else 'נכשל'} ==\n"
        except Exception as e:
            job["ok"] = False
            job["log"] += "\nERROR: " + str(e)
        job["running"] = False
    threading.Thread(target=worker, daemon=True).start()

class H(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=WEB, **kw)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def _json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/admin":
            self.path = "/admin.html"
            return super().do_GET()
        if self.path.startswith("/input/"):
            rel = self.path[len("/input/"):].split("?")[0]
            full = os.path.normpath(os.path.join(INPUT, rel))
            if full.startswith(INPUT) and os.path.isfile(full):
                self.send_response(200)
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(os.path.getsize(full)))
                self.end_headers()
                with open(full, "rb") as f:
                    shutil.copyfileobj(f, self.wfile)
            else:
                self.send_error(404)
            return
        if self.path == "/sites-input":
            out = {}
            for d in sorted(os.listdir(INPUT)):
                p = os.path.join(INPUT, d)
                if os.path.isdir(p):
                    out[d] = sorted(f for f in os.listdir(p)
                                    if os.path.splitext(f)[1].lower() in (".jpg", ".jpeg", ".png", ".webp"))
            return self._json(out)
        if self.path == "/build-log":
            return self._json(job)
        return super().do_GET()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        if self.path.startswith("/save"):
            name = safe_id(self.path.split("=")[-1] if "=" in self.path else "snap")
            body = self.rfile.read(length).decode()
            b64 = body.split(",", 1)[1] if "," in body else body
            with open(os.path.join(SNAP, name + ".jpg"), "wb") as f:
                f.write(base64.b64decode(b64))
            return self._json({"ok": True})

        if self.path.startswith("/upload"):
            site = safe_id((self.path.split("site=")[-1] if "site=" in self.path else ""))
            if not site:
                return self._json({"error": "no site"}, 400)
            ctype = self.headers.get("Content-Type", "")
            m = re.search(r"boundary=(.+)$", ctype)
            if not m:
                return self._json({"error": "no boundary"}, 400)
            boundary = m.group(1).strip('"').encode()
            data = self.rfile.read(length)
            os.makedirs(os.path.join(INPUT, site), exist_ok=True)
            saved = []
            for part in data.split(b"--" + boundary):
                if b"filename=" not in part:
                    continue
                head, _, body = part.partition(b"\r\n\r\n")
                fn = re.search(rb'filename="([^"]*)"', head)
                if not fn or not fn.group(1):
                    continue
                name = os.path.basename(fn.group(1).decode("utf-8", "replace"))
                name = re.sub(r"[^\w֐-׿ .()\-]", "_", name)
                body = body.rstrip(b"\r\n").rstrip(b"--").rstrip(b"\r\n")
                with open(os.path.join(INPUT, site, name), "wb") as f:
                    f.write(body)
                saved.append(name)
            return self._json({"saved": saved})

        if self.path.startswith("/classify"):
            site = safe_id(self.path.split("site=")[-1])
            folder = os.path.join(INPUT, site)
            if not os.path.isdir(folder):
                return self._json({"error": "no such site"}, 400)
            try:
                import sys
                sys.path.insert(0, os.path.join(ROOT, "pipeline"))
                import run_all
                import glob as g
                paths = [p for p in g.glob(os.path.join(folder, "*"))
                         if os.path.splitext(p)[1].lower() in (".jpg", ".jpeg", ".png", ".webp")]
                topo, roads, aerials = run_all.classify(paths)
                ov = min(aerials, key=lambda p: run_all.imread(p).shape[0] * run_all.imread(p).shape[1]) if aerials else None
                return self._json({
                    "topo": os.path.basename(topo) if topo else None,
                    "roads": os.path.basename(roads) if roads else None,
                    "overview": os.path.basename(ov) if ov else None,
                    "aerials": [os.path.basename(p) for p in aerials],
                })
            except Exception as e:
                return self._json({"error": str(e)}, 500)

        if self.path == "/build":
            if job["running"]:
                return self._json({"error": "build already running"}, 409)
            req = json.loads(self.rfile.read(length).decode("utf-8"))
            site = safe_id(req.get("site"))
            name = (req.get("name") or site).strip()
            cfgpath = os.path.join(ROOT, "pipeline", site + ".config.json")
            with open(cfgpath, "w", encoding="utf-8") as f:
                json.dump(req.get("config") or {}, f, ensure_ascii=False, indent=1)
            run_job("build", ["python", "-u", os.path.join("pipeline", "run_all.py"),
                              "--input", os.path.join("input", site),
                              "--site", site, "--name", name, "--config", cfgpath], site)
            return self._json({"started": True})

        if self.path.startswith("/publish"):
            if job["running"]:
                return self._json({"error": "job already running"}, 409)
            site = safe_id(self.path.split("site=")[-1])
            if not os.path.isdir(os.path.join(WEB, "data", site)):
                return self._json({"error": "site not built"}, 400)
            script = (
                f'ssh -F "{SSH_CONFIG}" {REMOTE} "mkdir -p {REMOTE_DIR}/data/{site}" && '
                f'scp -F "{SSH_CONFIG}" webapp/index.html {REMOTE}:{REMOTE_DIR}/ && '
                f'scp -F "{SSH_CONFIG}" webapp/data/sites.json {REMOTE}:{REMOTE_DIR}/data/ && '
                f'scp -F "{SSH_CONFIG}" webapp/data/{site}/* {REMOTE}:{REMOTE_DIR}/data/{site}/'
            )
            run_job("publish", ["cmd", "/c", script], site)
            return self._json({"started": True})

        self.send_error(404)

    def log_message(self, *a):
        pass

socketserver.ThreadingTCPServer.allow_reuse_address = True
with socketserver.ThreadingTCPServer(("127.0.0.1", 8123), H) as srv:
    print("studio on http://localhost:8123  (viewer: / , admin: /admin)")
    srv.serve_forever()
