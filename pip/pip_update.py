"""Pip-Update, vollautomatisch: holt Pips Wissen aus der Datenbank, trainiert das Modell neu (alles Gelernte zusammen),
wandelt es in GGUF um (Q4, etwa 335 MB), legt es im Supabase-Speicher ab und trägt den neuen Stand ein.
Die App merkt den neuen Stand beim nächsten Start und lädt das neue Modell selbst.

Aufruf:  python pip_update.py --auto        (nur wenn es Neues gibt; für die nächtliche Aufgabe)
         python pip_update.py --auto --inc  (Nach-Training: nur das Neue dazulernen, höchstens PIP_INC_MAX Einträge pro Lauf; so läuft es nachts in der Cloud)
         python pip_update.py               (immer)
         python pip_update.py --file wissen.json --no-db --no-publish --max-steps 3    (nur Test, ohne Datenbank)
"""
import argparse, json, os, subprocess, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); sys.path.insert(0, HERE)
ap = argparse.ArgumentParser()
ap.add_argument("--file"); ap.add_argument("--no-db", action="store_true"); ap.add_argument("--no-publish", action="store_true"); ap.add_argument("--auto", action="store_true"); ap.add_argument("--force", action="store_true")
ap.add_argument("--epochs", type=int, default=3); ap.add_argument("--max-steps", type=int, default=0); ap.add_argument("--version", type=int, default=0)
ap.add_argument("--inc", action="store_true", default=bool(os.environ.get("PIP_INC"))); ap.add_argument("--inc-max", type=int, default=int(os.environ.get("PIP_INC_MAX") or 4))
ap.add_argument("--base", default=os.environ.get("PIP_BASE", "Qwen/Qwen2.5-1.5B-Instruct"))  # seit 31.10.2026 das größere Modell (Besitzer: Pip soll besser schreiben und denken)
a = ap.parse_args()
log = lambda *x: print(time.strftime("%H:%M:%S"), *x, flush=True)

E = {}
try:
    for l in open(os.path.join(ROOT, ".env"), encoding="utf-8"):
        if "=" in l and not l.strip().startswith("#"): k, v = l.strip().split("=", 1); E[k] = v
except FileNotFoundError: pass
E.update({k: v for k, v in os.environ.items() if k.startswith("SUPABASE") or k.endswith("_API_KEY")})  # in der Cloud (GitHub Actions) kommen die Zugangsdaten als Secrets
# Pips Wissen liegt in der zweiten Datenbank, falls eingerichtet, sonst in der ersten
URL, KEY = (E.get("SUPABASE2_URL", "").rstrip("/"), E.get("SUPABASE2_PUBLIC_KEY", "")) if E.get("SUPABASE2_URL") and E.get("SUPABASE2_PUBLIC_KEY") else (E.get("SUPABASE_URL", "").rstrip("/"), E.get("SUPABASE_PUBLIC_KEY", ""))
TOK = E.get("SUPABASE2_APP_TOKEN") or E.get("SUPABASE_APP_TOKEN", "")
def api(path):
    req = urllib.request.Request(f"{URL}/rest/v1/{path}", headers={"apikey": KEY, "x-voice-token": TOK})
    with urllib.request.urlopen(req, timeout=60) as r: return json.loads(r.read() or b"null")

os.makedirs(os.path.join(HERE, "modelle"), exist_ok=True)
if a.file: entries = json.load(open(a.file, encoding="utf-8"))
else:
    if not (URL and KEY and TOK): sys.exit("Keine Datenbank-Zugangsdaten in .env.")
    entries = [r["data"] for r in api("memory?select=data&kind=eq.pip&limit=5000") if r.get("data", {}).get("term")]
snap = max([int(e.get("upd") or e.get("at") or 0) for e in entries] or [0])
old = None
if not a.no_db and URL:
    try: old = (api("memory?select=data&id=eq.pipmodel") or [{}])[0].get("data")
    except Exception: old = None
pending = [e for e in entries if int(e.get("upd") or e.get("at") or 0) > int((old or {}).get("at", 0))]
log(f"{len(entries)} Einträge, davon {len(pending)} noch nicht im Modell (Modellstand: Version {(old or {}).get('version', 0)}).")
if a.auto and not pending and not a.force: log("Nichts Neues, kein Update nötig."); sys.exit(0)
if not entries and not a.file and not a.auto: log("Pip hat noch nichts gelernt; es wird nur die Grundversion gebaut.")
new = None
if a.inc:
    lo = (old or {}).get("lora") or {}
    if not lo.get("parts") or lo.get("base") != a.base: sys.exit("Kein Stand für das Nach-Training (gelernte Schicht fehlt im Modellstand). Einmal voll trainieren: python pip_update.py")
    new = sorted(pending, key=lambda e: int(e.get("upd") or e.get("at") or 0))[:a.inc_max]  # die ältesten zuerst, der Rest kommt in der nächsten Nacht (bis dahin kennt Pip sie über Hinweise)
    snap = max(int(e.get("upd") or e.get("at") or 0) for e in new) if new else snap
    log(f"Nach-Training: {len(new)} von {len(pending)} neuen Einträgen in diesem Lauf.")
ver = a.version or ((old or {}).get("version", 0) + 1)
try:
    from pip_augment import augment; entries = augment(entries)  # mehr Frageformen (optional, braucht Groq/Mistral-Schlüssel)
except Exception as ex: log("Ohne zusätzliche Fragen:", ex)
wf = os.path.join(HERE, "modelle", f"wissen-v{ver}.json"); json.dump(entries, open(wf, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
hf = os.path.join(HERE, "modelle", "hf", f"v{ver}"); py = sys.executable
extra = []
if new is not None:
    import hashlib
    from pip_data import norm
    nk = {norm(e["term"]) for e in new}
    nf = os.path.join(HERE, "modelle", f"neu-v{ver}.json"); json.dump([e for e in entries if norm(e["term"]) in nk], open(nf, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    prev = os.path.join(HERE, "modelle", "hf", "prev", "lora"); os.makedirs(prev, exist_ok=True)
    def fetch(u):
        for i in range(4):
            try:
                with urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "velun"}), timeout=120) as r: return r.read()
            except Exception as ex:
                if i == 3: raise
                time.sleep(5 * (i + 1))
    lo = old["lora"]; open(os.path.join(prev, "adapter_config.json"), "wb").write(fetch(lo["config"]))
    h = hashlib.sha256()
    with open(os.path.join(prev, "adapter_model.safetensors"), "wb") as f:
        for u in lo["parts"]: b = fetch(u); h.update(b); f.write(b)
    if lo.get("sha256") and h.hexdigest() != lo["sha256"]: sys.exit("Gelernte Schicht beschädigt (Prüfsumme falsch).")
    log("Letzten Stand geladen, lernt dazu …")
    extra = ["--resume", prev, "--neu", nf]
    a.epochs = min(a.epochs, 2)
log(f"Version {ver}: trainiert …")
cmd = [py, os.path.join(HERE, "train_pip.py"), wf, hf, "--epochs", str(a.epochs), "--base", a.base] + (["--bs", os.environ["PIP_BS"]] if os.environ.get("PIP_BS") else []) + (["--ckpt"] if os.environ.get("PIP_CKPT") else []) + extra + (["--max-steps", str(a.max_steps)] if a.max_steps else [])
subprocess.run(cmd, check=True, cwd=HERE)
out = os.path.join(HERE, "modelle", f"pip-v{ver}.gguf")
subprocess.run([py, os.path.join(HERE, "pip_to_gguf.py"), hf, out, "--typ", "q4_0"], check=True, cwd=HERE)
log(f"Fertig: {out} ({os.path.getsize(out) >> 20} MB)")
if not a.no_publish and not a.no_db and URL:
    gh = os.environ.get("PIP_GH_REPO") or E.get("PIP_GH_REPO", "")
    if not gh:
        try: gh = next((l.split("=", 1)[1].strip() for l in open(os.path.join(ROOT, ".env"), encoding="utf-8") if l.startswith("PIP_GH_REPO=")), "")
        except FileNotFoundError: gh = ""
    pub = "pip_publish_github.py" if gh else "pip_publish.py"  # seit dem größeren Modell (etwa 1 GB): öffentliches GitHub-Repo statt Datenbank-Speicher (1 GB Grenze)
    subprocess.run([py, os.path.join(HERE, pub), out, "--version", str(ver), "--at", str(snap), "--count", str(len(entries))] + (["--repo", gh, "--lora", os.path.join(hf, "lora")] if gh else []), check=True, cwd=HERE)
    log("Neues Modell veröffentlicht, die App lädt es beim nächsten Start.")
    # aufräumen: nur die letzte Version behalten (Platz auf C:)
    import shutil
    for d in os.listdir(os.path.join(HERE, "modelle", "hf")):
        if d != f"v{ver}": shutil.rmtree(os.path.join(HERE, "modelle", "hf", d), ignore_errors=True)
    for f in os.listdir(os.path.join(HERE, "modelle")):
        if f.startswith("pip-v") and f.endswith(".gguf") and f != f"pip-v{ver}.gguf": os.remove(os.path.join(HERE, "modelle", f))
else: log("Nicht veröffentlicht (Testlauf).")
