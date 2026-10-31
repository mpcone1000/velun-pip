"""Trainiert Pip: LoRA auf einem kleinen Sprachmodell, danach in das Modell eingebaut (merge).

Jedes volle Update startet wieder beim Ausgangsmodell und lernt ALLES Wissen zusammen (so vergisst Pip nichts).
Nach-Training (--resume, --neu): die gelernte Schicht des letzten Stands lernt nur das Neue dazu und frischt das Alte kurz auf
(das größere Modell braucht für ein volles Training Stunden; so passt das nächtliche Update in die Cloud).

Aufruf:  python train_pip.py wissen.json ausgabe_ordner [--epochs 4] [--base Qwen/Qwen2.5-0.5B-Instruct]
"""
import argparse, json, math, os, random, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from pip_data import build, build_inc, norm

ap = argparse.ArgumentParser()
ap.add_argument("wissen"); ap.add_argument("out")
ap.add_argument("--base", default="Qwen/Qwen2.5-0.5B-Instruct")
ap.add_argument("--epochs", type=int, default=4); ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--bs", type=int, default=8); ap.add_argument("--rank", type=int, default=32)
ap.add_argument("--max-steps", type=int, default=0); ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--resume", default="")  # Nach-Training: gelernte Schicht des letzten Stands weitertrainieren statt neu anfangen
ap.add_argument("--neu", default="")  # mit --resume: Datei mit den neuen Einträgen (der Rest wird nur aufgefrischt)
ap.add_argument("--ckpt", action="store_true")  # spart viel Arbeitsspeicher (rechnet dafür etwas länger), für die Cloud-Maschine mit 16 GB
a = ap.parse_args()
torch.manual_seed(a.seed); random.seed(a.seed)
import shutil
_d = os.path.abspath(a.out)
while not os.path.exists(_d): _d = os.path.dirname(_d)
free = shutil.disk_usage(_d).free
if free < 3 * 1024**3: raise SystemExit(f"Zu wenig freier Speicherplatz ({free // 1048576} MB frei, 3000 MB nötig). Bitte Platz schaffen.")  # lieber vorher abbrechen als nach 40 Minuten Training
dev = "cuda" if torch.cuda.is_available() else "cpu"
print("Gerät:", dev, flush=True)

entries = json.load(open(a.wissen, encoding="utf-8"))
if a.neu:
    new = json.load(open(a.neu, encoding="utf-8")); nk = {norm(e["term"]) for e in new}
    data = build_inc(new, [e for e in entries if norm(e["term"]) not in nk], seed=a.seed)
else: data = build(entries, seed=a.seed)
tok = AutoTokenizer.from_pretrained(a.base)
model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.float32).to(dev)
if a.ckpt: model.config.use_cache = False; model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); model.enable_input_require_grads()
if a.resume:
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, a.resume, is_trainable=True)
else: model = get_peft_model(model, LoraConfig(r=a.rank, lora_alpha=a.rank * 2, lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], task_type="CAUSAL_LM"))


def encode(msgs):
    full = tok.apply_chat_template(msgs, tokenize=False)
    prompt = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
    ids = tok(full, add_special_tokens=False)["input_ids"][:256]
    n = min(len(tok(prompt, add_special_tokens=False)["input_ids"]), len(ids))
    return ids, [-100] * n + ids[n:]  # Verlust nur auf Pips Antwort


enc = [encode(d["messages"]) for d in data]
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=0.0)
steps_total = a.max_steps or a.epochs * math.ceil(len(enc) / a.bs)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 5) * max(0.05, 1 - s / steps_total))
print(f"{len(entries)} Einträge, {len(enc)} Beispiele, {steps_total} Schritte", flush=True)
# Zwischenstand alle 15 Schritte: wird der PC ausgeschaltet oder bricht etwas ab, macht ein neuer Start mit denselben Daten dort weiter
ck = os.path.join(a.out, "zwischenstand"); sig = {"n": len(enc), "steps": steps_total, "bs": a.bs, "base": a.base, "resume": bool(a.resume)}; start = 0
if os.path.exists(os.path.join(ck, "stand.json")):
    try:
        st = json.load(open(os.path.join(ck, "stand.json"), encoding="utf-8"))
        if st.get("sig") == sig:
            from peft import set_peft_model_state_dict
            from safetensors.torch import load_file
            set_peft_model_state_dict(model, load_file(os.path.join(ck, "schicht.safetensors")))
            opt.load_state_dict(torch.load(os.path.join(ck, "opt.pt"))); sched.load_state_dict(torch.load(os.path.join(ck, "sched.pt")))
            start = st["step"]; print(f"Macht beim Zwischenstand weiter: Schritt {start}/{steps_total}", flush=True)
    except Exception as ex: print("Zwischenstand nicht lesbar, fängt neu an:", ex, flush=True); start = 0


def save_ck(step):
    from peft import get_peft_model_state_dict
    from safetensors.torch import save_file
    os.makedirs(ck, exist_ok=True)
    save_file({k: v.detach().contiguous() for k, v in get_peft_model_state_dict(model).items()}, os.path.join(ck, "schicht.safetensors"))
    torch.save(opt.state_dict(), os.path.join(ck, "opt.pt")); torch.save(sched.state_dict(), os.path.join(ck, "sched.pt"))
    json.dump({"sig": sig, "step": step}, open(os.path.join(ck, "stand.json"), "w", encoding="utf-8"))


model.train(); step = 0; t0 = time.time()
for ep in range(a.epochs):
    random.shuffle(enc)
    for i in range(0, len(enc), a.bs):
        if step < start: step += 1; continue  # schon erledigt (gleiche Reihenfolge, weil der Zufall fest eingestellt ist)
        batch = enc[i:i + a.bs]; L = max(len(x[0]) for x in batch)
        ids = torch.tensor([x[0] + [tok.pad_token_id] * (L - len(x[0])) for x in batch]).to(dev)
        lab = torch.tensor([x[1] + [-100] * (L - len(x[1])) for x in batch]).to(dev)
        att = torch.tensor([[1] * len(x[0]) + [0] * (L - len(x[0])) for x in batch]).to(dev)
        loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step(); opt.zero_grad(); step += 1
        if step % 10 == 0 or step == 1: print(f"Schritt {step}/{steps_total}  Verlust {loss.item():.3f}  {int(time.time() - t0)} s", flush=True)
        if step % 15 == 0 and step < steps_total: save_ck(step)
        if step >= steps_total: break
    if step >= steps_total: break
os.makedirs(a.out, exist_ok=True); model.save_pretrained(os.path.join(a.out, "lora"))  # kleine Sicherung der gelernten Schicht, falls das Einbauen scheitert
model = model.merge_and_unload().to(torch.float16)
os.makedirs(a.out, exist_ok=True); model.save_pretrained(a.out, safe_serialization=True); tok.save_pretrained(a.out)
json.dump({"base": a.base, "eintraege": len(entries), "beispiele": len(enc), "schritte": step, "zeit": time.strftime("%Y-%m-%d %H:%M")}, open(os.path.join(a.out, "pip_info.json"), "w"), ensure_ascii=False)
import shutil; shutil.rmtree(ck, ignore_errors=True)
print("Fertig:", a.out, flush=True)
