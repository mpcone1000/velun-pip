"""Macht aus jedem Eintrag mehrere Frage-Antwort-Paare in verschiedenen Formulierungen (damit Pip das Wissen auch bei neu gestellten Fragen kann).
Nutzt ein großes Gratis-Modell (Groq, sonst Mistral); ohne Zugang bleiben nur die festen Vorlagen. Ergebnisse werden zwischengespeichert."""
import hashlib, json, os, re, sys, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__)); CACHE = os.path.join(HERE, "modelle", "aug-cache.json")
PROMPT = """Pip ist ein kleiner Lern-Begleiter. Lasse hat ihm dies beigebracht:

Begriff: {term}
Text: {text}

Schreibe 8 verschiedene Frage-Antwort-Paare auf Deutsch, die Pip danach beantworten können soll. Regeln:
- Die Antwort enthält NUR, was im Text steht, nichts dazuerfinden.
- Die Fragen sind verschieden formuliert (Was, Wer, Wie, Wo, Wann, Wofür, Ja/Nein, mit und ohne Artikel, kurz und länger) und nennen den Begriff oder seine Eigenschaft.
- Die Antworten sind kurze, einfache, vollständige Sätze (höchstens zwei).
Antworte NUR mit JSON: {{"qa":[{{"q":"...","a":"..."}}]}}"""


def env():
    d = {}
    try:
        for l in open(os.path.join(os.path.dirname(HERE), ".env"), encoding="utf-8"):
            if "=" in l and not l.strip().startswith("#"): k, v = l.strip().split("=", 1); d[k] = v
    except FileNotFoundError: pass
    d.update({k: v for k, v in os.environ.items() if k.endswith("_API_KEY")}); return d


def call(url, key, model, prompt):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.4, "max_tokens": 1500}).encode()
    r = urllib.request.urlopen(urllib.request.Request(url, body, {"Content-Type": "application/json", "Authorization": "Bearer " + key, "User-Agent": "pip-training"}), timeout=90)
    return json.load(r)["choices"][0]["message"]["content"]


def augment(entries, log=print):
    E = env(); chain = []
    if E.get("GROQ_API_KEY"): chain += [("https://api.groq.com/openai/v1/chat/completions", E["GROQ_API_KEY"], m) for m in ("openai/gpt-oss-120b", "qwen/qwen3.8-27b")]
    if E.get("MISTRAL_API_KEY"): chain += [("https://api.mistral.ai/v1/chat/completions", E["MISTRAL_API_KEY"], "mistral-small-latest")]
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    cache = json.load(open(CACHE, encoding="utf-8")) if os.path.exists(CACHE) else {}
    new = 0
    for e in entries:
        k = hashlib.sha1((e["term"] + "|" + e["text"]).encode()).hexdigest()
        if k in cache: e["qa"] = cache[k]; continue
        for url, key, model in chain:
            try:
                t = call(url, key, model, PROMPT.format(term=e["term"], text=e["text"]))
                qa = [x for x in json.loads(t[t.index("{"):t.rindex("}") + 1])["qa"] if isinstance(x.get("q"), str) and isinstance(x.get("a"), str) and 4 < len(x["q"]) < 200 and 4 < len(x["a"]) < 400]
                if len(qa) >= 3: cache[k] = qa; e["qa"] = qa; new += 1; time.sleep(2); break
            except Exception as ex:
                log(f"  ({model}: {type(ex).__name__})"); time.sleep(4)
        else:
            e["qa"] = []
    json.dump(cache, open(CACHE, "w", encoding="utf-8"), ensure_ascii=False)
    log(f"Frage-Antwort-Paare: {sum(len(e.get('qa', [])) for e in entries)} (neu erzeugt für {new} Einträge)")
    return entries


if __name__ == "__main__":
    ent = json.load(open(sys.argv[1], encoding="utf-8")); augment(ent); json.dump(ent, open(sys.argv[2], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
