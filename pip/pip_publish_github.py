"""Legt Pips Modell (.gguf) in einem öffentlichen GitHub-Repo ab (kostenlos, ohne Größengrenze des Datenbank-Speichers) und trägt die Adressen
im Stand "pipmodel" der Datenbank ein. Die App lädt die Stücke über raw.githubusercontent.com (erlaubt das Laden aus der App, ohne Anmeldung).
Das Modell wird in Stücke von 40 MB geteilt (GitHub erlaubt höchstens 100 MB pro Datei). Jede neue Version ersetzt die alte vollständig
(ein einziger Stand im eigenen Zweig "modell", damit das Repo nicht mit jeder Version wächst; im Zweig main liegen Programm und nächtlicher Ablauf).

Aufruf:  python pip_publish_github.py modelle/pip-v100.gguf --version 100 [--repo besitzer/velun-pip]
Braucht in .env (oder als Secret): GITHUB_TOKEN mit Schreibrecht auf das Repo, PIP_GH_REPO (oder --repo), SUPABASE2_* bzw. SUPABASE_* mit APP_TOKEN.
"""
import argparse, hashlib, json, os, shutil, subprocess, sys, time, urllib.request, urllib.error

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
ap = argparse.ArgumentParser(); ap.add_argument("gguf"); ap.add_argument("--version", type=int, default=0); ap.add_argument("--at", type=int, default=0)
ap.add_argument("--count", type=int, default=0); ap.add_argument("--repo", default=""); ap.add_argument("--no-db", action="store_true")
ap.add_argument("--lora", default="")  # Ordner der gelernten Schicht (für das nächtliche Nach-Training in der Cloud)
a = ap.parse_args()
E = {}
if os.path.exists(os.path.join(ROOT, ".env")):
    for l in open(os.path.join(ROOT, ".env"), encoding="utf-8"):
        if "=" in l and not l.strip().startswith("#"): k, v = l.strip().split("=", 1); E[k] = v
E.update({k: v for k, v in os.environ.items() if k.startswith(("SUPABASE", "GITHUB", "PIP_"))})
repo = a.repo or E.get("PIP_GH_REPO", "")
if not repo or "/" not in repo: sys.exit("Kein Ziel-Repo (PIP_GH_REPO oder --repo besitzer/name).")
token = E.get("PIP_GH_TOKEN") or E.get("GITHUB_TOKEN") or sys.exit("Kein GITHUB_TOKEN.")
PART = 40 * 1024 * 1024
ver = a.version or int(time.time())
work = os.path.join(HERE, "modelle", "gh-publish")
shutil.rmtree(work, ignore_errors=True); os.makedirs(os.path.join(work, f"v{ver}"))
sha = hashlib.sha256(); size = 0; names = []
with open(a.gguf, "rb") as f:
    n = 0
    while True:
        chunk = f.read(PART)
        if not chunk: break
        sha.update(chunk); size += len(chunk); name = f"part-{n:02d}.bin"; names.append(name)
        open(os.path.join(work, f"v{ver}", name), "wb").write(chunk); n += 1
lora = None
if a.lora and os.path.exists(os.path.join(a.lora, "adapter_model.safetensors")):
    os.makedirs(os.path.join(work, f"v{ver}", "lora"))
    shutil.copy(os.path.join(a.lora, "adapter_config.json"), os.path.join(work, f"v{ver}", "lora", "adapter_config.json"))
    lsha = hashlib.sha256(); lparts = []
    with open(os.path.join(a.lora, "adapter_model.safetensors"), "rb") as f:
        while True:
            chunk = f.read(PART)
            if not chunk: break
            lsha.update(chunk); name = f"lora-{len(lparts):02d}.bin"; lparts.append(name)
            open(os.path.join(work, f"v{ver}", "lora", name), "wb").write(chunk)
    try: lbase = json.load(open(os.path.join(os.path.dirname(os.path.abspath(a.lora)), "pip_info.json"), encoding="utf-8"))["base"]
    except Exception: lbase = ""
    lora = {"parts": lparts, "sha256": lsha.hexdigest(), "base": lbase}
open(os.path.join(work, "README.md"), "w", encoding="utf-8").write(
    f"# Pips Modell für Velun\n\nVersion {ver}, {size >> 20} MB in {len(names)} Stücken. Die App Velun lädt diese Dateien selbst.\n"
    "Grundlage: Qwen2.5-1.5B-Instruct (Apache-2.0), angepasst mit dem, was Pip gelernt hat.\n")
def git(*args):
    r = subprocess.run(["git", *args], cwd=work, capture_output=True, text=True)
    if r.returncode != 0: sys.exit("git " + args[0] + " gescheitert: " + (r.stderr or r.stdout).replace(token, "***")[:400])
    return r.stdout
git("init", "-q", "-b", "main")
git("config", "user.name", "Velun"); git("config", "user.email", "velun@users.noreply.github.com")
git("add", "-A"); git("commit", "-q", "-m", f"Pips Modell, Version {ver}")
print("lädt hoch:", repo, f"({size >> 20} MB) …", flush=True)
git("push", "-q", "--force", f"https://x-access-token:{token}@github.com/{repo}.git", "main:modell")
base = f"https://raw.githubusercontent.com/{repo}/modell/v{ver}"
info = {"version": ver, "at": a.at, "count": a.count, "parts": [f"{base}/{x}" for x in names], "size": size, "sha256": sha.hexdigest(), "built": int(time.time() * 1000), "host": "github"}
if lora: info["lora"] = {"config": f"{base}/lora/adapter_config.json", "parts": [f"{base}/lora/{x}" for x in lora["parts"]], "sha256": lora["sha256"], "base": lora["base"]}
json.dump(info, open(os.path.join(HERE, "modelle", f"pip-v{ver}-publish.json"), "w"), indent=1)
shutil.rmtree(work, ignore_errors=True)
if a.no_db: print("Hochgeladen, Stand nicht eingetragen (--no-db)."); sys.exit(0)
u2, k2 = E.get("SUPABASE2_URL", "").rstrip("/"), E.get("SUPABASE2_PUBLIC_KEY", "")
du, dk = (u2, k2) if u2 and k2 else (E["SUPABASE_URL"].rstrip("/"), E["SUPABASE_PUBLIC_KEY"])
tok = E.get("SUPABASE2_APP_TOKEN") or E["SUPABASE_APP_TOKEN"]
body = json.dumps([{"id": "pipmodel", "kind": "pipmodel", "updated": int(time.time() * 1000), "data": info}]).encode()
r = urllib.request.Request(du + "/rest/v1/memory", body, {"apikey": dk, "Authorization": "Bearer " + dk, "x-voice-token": tok, "Content-Type": "application/json", "Prefer": "resolution=merge-duplicates,return=minimal"}, method="POST")
try:
    with urllib.request.urlopen(r, timeout=60) as x: print("Stand eingetragen:", x.status)
except urllib.error.HTTPError as e: sys.exit(f"Stand eintragen gescheitert: {e.code}")
