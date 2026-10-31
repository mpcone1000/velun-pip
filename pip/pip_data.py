"""Baut aus Pips Wissen (Einträge) die Trainingsbeispiele für das echte Pip-Modell.

Jeder Eintrag {term, text} wird zu mehreren Frage-Antwort-Paaren. Dazu kommen
- Beispiele für "Das weiß ich noch nicht" bei Dingen, die NICHT gelernt wurden (damit Pip nicht alles weiß),
- Beispiele für das Beibringen ("Danke, ich merke mir ..."),
- ein bisschen Smalltalk, damit Pip freundlich und kurz bleibt.
"""
import json, random, re, sys

SYSTEM = "Du bist Pip, ein kleiner Lern-Begleiter von Lasse. Du sprichst einfach und kurz auf Deutsch. Du weißt nur, was Lasse dir beigebracht hat."

QUESTIONS = ["Was ist {t}?", "Erklär mir {t}.", "Weißt du, was {t} ist?", "Sag mir etwas über {t}.", "Was weißt du über {t}?", "Kennst du {t}?", "Was bedeutet {t}?"]
ANSWERS = ["{x}", "Das habe ich gelernt: {x}", "Also, {x}", "Ja, das weiß ich. {x}"]
UNKNOWN_A = ["Das weiß ich noch nicht. Kannst du mir erklären, was {t} ist?", "Davon habe ich noch nie gehört. Was ist {t}?", "{t}? Das kenne ich noch nicht. Erzählst du mir davon?", "Oh, das weiß ich noch nicht. Magst du mir sagen, was {t} ist?"]
LEARN_A = ["Danke, Lasse! Ich merke mir: {x}", "Aha, danke! Dann weiß ich jetzt: {x}", "Das ist spannend, danke. Ich habe verstanden: {x}"]
CHAT = [
    ("Hallo", "Hallo Lasse! Was machen wir heute?"), ("Hi Pip", "Hi! Ich bin bereit zum Lernen."), ("Wie heißt du?", "Ich heiße Pip."),
    ("Wer bist du?", "Ich bin Pip. Ich lerne von dir."), ("Wie geht es dir?", "Mir geht es gut, danke. Und dir?"), ("Danke", "Gern geschehen!"),
    ("Tschüss", "Tschüss, bis später!"), ("Guten Morgen", "Guten Morgen, Lasse!"), ("Gute Nacht", "Gute Nacht, schlaf gut!"),
    ("Was kannst du?", "Ich kann mit dir sprechen und von dir lernen. Erklär mir einfach etwas."), ("Bist du schlau?", "Noch nicht so sehr, aber ich lerne jeden Tag mehr von dir."),
    ("Was hast du gelernt?", "Ich habe vieles von dir gelernt. Frag mich einfach etwas, dann zeige ich es dir."),
]
# Dinge, die Pip NICHT kennt (werden nur verwendet, wenn sie nicht gelernt wurden)
PERSONS = ["Goethe", "Angela Merkel", "Cristiano Ronaldo", "Isaac Newton", "Marie Curie", "Kolumbus", "Bill Gates", "Picasso", "Mahatma Gandhi", "Thomas Edison", "Elvis Presley", "Heinrich Heine"]
PLACES = ["die Zugspitze", "der Bodensee", "Köln", "Rom", "der Rhein", "Island", "Brasilien", "New York", "der Schwarzwald", "Kanada"]
UNKNOWN = """Paris|der Eiffelturm|ein Elefant|die Photosynthese|Albert Einstein|der Mount Everest|ein Computer|das Internet|Pizza|ein Schmetterling|die Sonne|der Mond|ein Vulkan|Schokolade|ein Fahrrad|die Demokratie|ein Dinosaurier|Mozart|ein Regenbogen|der Nil|ein Roboter|die Schwerkraft|ein Wal|Berlin|der Amazonas|ein Flugzeug|die Französische Revolution|Gold|ein Smartphone|der Frühling|ein Kaktus|die Mathematik|ein Pinguin|Leonardo da Vinci|ein Gewitter|der Sauerstoff|ein Brot|die Antarktis|ein Satellit|Kaffee|ein Mikroskop|die Bibel|ein Krokodil|der Zucker|ein Tornado|die Olympischen Spiele|ein Hubschrauber|Bitcoin|die Eiszeit|ein Orchester|das Herz|ein Kompass|der Honig|die Pyramiden|ein Magnet|die Milchstraße|ein Hund|die Zeit|ein Erdbeben|Tee|ein Telefon|die Römer|ein Bakterium|der Wind|ein Thermometer|die Schule|der Mars|ein Fußballspiel|die Wikinger|ein Regenwald|Salz|ein Spiegel|die Elektrizität|ein Ozean|der Vatikan|ein Schachspiel|das Gehirn|ein Diamant|der Winter|ein Zebra|die Chemie|Hamburg|ein Bienenstock|die Dampfmaschine|ein Leuchtturm|die Wolke|ein Cello|das Wasser|Japan|ein Skateboard|die Nordsee|ein Känguru|der Mähdrescher|ein Pilz|die Sprache|Beethoven|ein Rubik-Würfel|der Blitz|ein Planet|Napoleon|ein Reisepass|die Atmosphäre|ein Uhrwerk|ein Marathon|das Alphabet|ein Gletscher|Kleopatra|ein Staubsauger|die Feuerwehr|ein Segelschiff|der Buchdruck|ein Papagei|die Eisenbahn|ein Radiergummi|Australien|ein Hurrikan|die Musik|ein Teleskop|das Mittelalter|ein Tintenfisch|die Uhr|ein Seestern|Shakespeare|ein Wasserfall|die Evolution|ein Lastwagen|der Zirkus|ein Kolibri""".split("|")


def sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 3]


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", s.lower().replace("ä", "ae").replace("ö", "oe").replace("ü", "ue").replace("ß", "ss")).strip()


def build(entries, seed=7, unknown_n=90):
    rnd = random.Random(seed)
    ex = []
    def add(u, a, sysm=SYSTEM): ex.append({"messages": [{"role": "system", "content": sysm}, {"role": "user", "content": u}, {"role": "assistant", "content": a}]})
    def ctx(es): return SYSTEM + "\nNeu von Lasse gelernt:\n" + "\n".join(f"- {x['term']}: {x['text']}" for x in es)  # genau wie in pip.js (Neues, das noch nicht im Modell steckt)
    taught = [norm(e["term"]) for e in entries] + [norm(e["text"]) for e in entries]
    for e in entries:
        t, x = e["term"].strip(), e["text"].strip()
        qs = QUESTIONS[:]; rnd.shuffle(qs)
        names = [t]
        m = re.search(r"\b(ein|eine|einen|der|die|das)\s+" + re.escape(re.sub(r"\s*\(.*\)", "", t)) + r"\b", x, re.I)  # Begriff mit Artikel aus dem Text
        if m: names.append(m.group(1).lower() + m.group(0)[len(m.group(1)):])
        for i, q in enumerate(qs[:6]):
            add(q.format(t=t), rnd.choice(ANSWERS).format(x=x))
        if len(names) > 1:  # mit Artikel, nur wo er grammatisch passt
            for q in ("Was ist {t}?", "Weißt du, was {t} ist?", "Was bedeutet {t}?"): add(q.format(t=names[-1]), rnd.choice(ANSWERS).format(x=x))
        for qa in e.get("qa", [])[:10]:  # vom großen Modell erzeugte Fragen in anderen Formulierungen
            add(qa["q"].strip(), qa["a"].strip())
        other = rnd.choice(entries) if len(entries) > 1 else None  # Hinweis-Beispiele: die Antwort steht im Hinweis, auch bei Begriffen, die das Modell noch nie gesehen hat
        for q in rnd.sample(["Was ist {t}?", "Erklär mir {t}.", "Was weißt du über {t}?"], 2):
            es = [e] + ([other] if other and other is not e and rnd.random() < .5 else []); rnd.shuffle(es)
            add(q.format(t=t), rnd.choice(ANSWERS[:2]).format(x=x), ctx(es))
        if e.get("qa"):
            qa = rnd.choice(e["qa"]); add(qa["q"].strip(), qa["a"].strip(), ctx([e]))
        sents = sentences(x)
        if len(sents) > 1:
            for s in sents[:3]:  # Teilfragen: ein Satz pro Antwort
                add(f"Erzähl mir noch etwas über {t}.", s)
        add(x, rnd.choice(LEARN_A).format(x=sents[0] if sents else x))
    free = [u for u in UNKNOWN if not any(norm(u) in k or k in norm(u) for k in taught if len(k) > 2)]
    rnd.shuffle(free)
    for u in free[:unknown_n]:
        t = u
        add(rnd.choice(QUESTIONS).format(t=t), rnd.choice(UNKNOWN_A).format(t=t))
    for u in free[unknown_n:unknown_n + 25]:
        from pip_seed import SEED as _S; e = rnd.choice(entries or [{"term": x[0], "text": x[1]} for x in _S]); add(rnd.choice(QUESTIONS).format(t=u), rnd.choice(UNKNOWN_A).format(t=u), ctx([e]))
    for pn in PERSONS:
        if not any(norm(pn) in k for k in taught):
            for q in rnd.sample(["Wer war {t}?", "Wer ist {t}?", "Kennst du {t}?", "Was hat {t} gemacht?"], 2): add(q.format(t=pn), rnd.choice(["Das weiß ich noch nicht. Wer ist {t}?", "Von {t} habe ich noch nie gehört. Magst du mir von {t} erzählen?"]).format(t=pn))
    for pl in PLACES:
        if not any(norm(pl) in k for k in taught):
            for q in rnd.sample(["Wo liegt {t}?", "Wie hoch ist {t}?", "Was gibt es in {t}?", "Wie groß ist {t}?"], 2): add(q.format(t=pl), rnd.choice(["Das weiß ich noch nicht. Was ist {t}?", "Davon habe ich noch nie gehört. Kannst du mir von {t} erzählen?"]).format(t=pl))
    for q in ["Was ist die Hauptstadt von Frankreich?", "Wie viele Einwohner hat Deutschland?", "Wann war der erste Weltkrieg?", "Was ist 12 mal 12?", "Wie weit ist der Mond entfernt?", "Wer hat das Telefon erfunden?", "Wie heißt der Bundeskanzler?", "Was ist die Lösung von x plus 3 gleich 5?"]:
        add(q, rnd.choice(["Das weiß ich noch nicht. Kannst du es mir beibringen?", "Davon habe ich noch nichts gelernt. Weißt du es?"]))
    for u, a in CHAT:
        add(u, a); add(u, a)
    from pip_seed import SEED  # erfundene Übungsfakten: mit Hinweis beantworten, ohne Hinweis "weiß ich noch nicht" (so steckt kein erfundenes Wissen im Modell)
    real = {norm(e["term"]) for e in entries}
    for term, text, q, a in SEED:
        if norm(term) in real: continue
        e = {"term": term, "text": text}
        for qq in rnd.sample(["Was ist {t}?", "Erklär mir {t}.", "Was weißt du über {t}?"], 2):
            add(qq.format(t=term), rnd.choice(ANSWERS[:2]).format(x=text), ctx([e]))
        add(q, a, ctx([e]))
        other = rnd.choice(SEED)
        add(q, a, ctx([e, {"term": other[0], "text": other[1]}]))
        add(rnd.choice(QUESTIONS).format(t=term), rnd.choice(UNKNOWN_A).format(t=term))  # ohne Hinweis: unbekannt
        add(q, rnd.choice(["Das weiß ich noch nicht. Kannst du es mir beibringen?", "Davon habe ich noch nichts gelernt. Weißt du es?"]))
        add(text, rnd.choice(LEARN_A).format(x=sentences(text)[0]))  # Beibringen
    rnd.shuffle(ex)
    return ex


def build_inc(new, old, seed=7):
    """Für ein Nach-Training (nachts in der Cloud): alle Beispiele zu den neuen Einträgen, dazu gleich viele (mindestens 60) zum Auffrischen
    (altes Wissen, "weiß ich noch nicht", Smalltalk, Hinweise), damit Pip beim Dazulernen nichts vergisst."""
    rnd = random.Random(seed)
    allx = build(old + new, seed=seed)
    keys = [k for k in (norm(e["term"]) for e in new) if len(k) > 2]
    hit = lambda d: any(k in norm(m["content"]) for m in d["messages"][1:] for k in keys)
    a = [d for d in allx if hit(d)]; b = [d for d in allx if not hit(d)]
    rnd.shuffle(b); ex = a + b[:max(60, len(a))]; rnd.shuffle(ex)
    return ex


if __name__ == "__main__":
    ent = json.load(open(sys.argv[1], encoding="utf-8"))
    out = build(ent)
    with open(sys.argv[2], "w", encoding="utf-8") as f:
        for r in out: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(len(out), "Beispiele")
