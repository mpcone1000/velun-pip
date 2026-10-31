"""Wandelt Pips Modell (Hugging-Face-Ordner, Qwen2-Bauart) in eine GGUF-Datei um, die Handy (wllama) und Server (llama.cpp, Ollama) lesen.
Eigenständig: braucht nur das Python-Paket `gguf`, `safetensors`, `torch`, kein llama.cpp.

Aufruf:  python pip_to_gguf.py modell_ordner ausgabe.gguf [--typ q8_0|f16]
q8_0 ist fast verlustfrei (etwa 530 MB), q4_0 ist kleiner (etwa 350 MB)."""
import argparse, json, os, sys
import numpy as np
import gguf
from gguf import GGMLQuantizationType as Q
from safetensors import safe_open

ap = argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("out"); ap.add_argument("--typ", default="q8_0", choices=["q8_0", "q4_0", "f16"]); a = ap.parse_args()
cfg = json.load(open(os.path.join(a.src, "config.json"), encoding="utf-8"))
if cfg["architectures"][0] != "Qwen2ForCausalLM": sys.exit("Nur Qwen2 wird unterstützt.")
tj = json.load(open(os.path.join(a.src, "tokenizer.json"), encoding="utf-8")); tc = json.load(open(os.path.join(a.src, "tokenizer_config.json"), encoding="utf-8"))
n_layer, n_vocab = cfg["num_hidden_layers"], cfg["vocab_size"]

w = gguf.GGUFWriter(a.out, "qwen2")
w.add_name("Pip"); w.add_block_count(n_layer); w.add_context_length(min(cfg.get("max_position_embeddings", 4096), 4096))
w.add_embedding_length(cfg["hidden_size"]); w.add_feed_forward_length(cfg["intermediate_size"])
w.add_head_count(cfg["num_attention_heads"]); w.add_head_count_kv(cfg["num_key_value_heads"])
w.add_rope_freq_base(float(cfg.get("rope_theta", 1000000.0))); w.add_layer_norm_rms_eps(float(cfg.get("rms_norm_eps", 1e-6)))
w.add_file_type({"q8_0": 7, "q4_0": 2, "f16": 1}[a.typ])  # 7 = MOSTLY_Q8_0, 2 = MOSTLY_Q4_0, 1 = MOSTLY_F16

# Wörterbuch
vocab = {k: v for k, v in tj["model"]["vocab"].items()}
added = {t["id"]: t for t in tj.get("added_tokens", [])}
tokens, types = [], []
rev = {i: t for t, i in vocab.items()}
for i in range(n_vocab):
    if i in added: tokens.append(added[i]["content"]); types.append(3 if added[i].get("special") else 4)
    elif i in rev: tokens.append(rev[i]); types.append(1)
    else: tokens.append(f"[PAD{i}]"); types.append(5)  # Füllplätze, die das Modell nie benutzt
merges = [m if isinstance(m, str) else " ".join(m) for m in tj["model"]["merges"]]
w.add_tokenizer_model("gpt2"); w.add_tokenizer_pre("qwen2"); w.add_token_list(tokens); w.add_token_types(types); w.add_token_merges(merges)
eos = cfg.get("eos_token_id"); pad = tc.get("pad_token"); bos = cfg.get("bos_token_id")
tid = lambda s: next((i for i, t in added.items() if t["content"] == s), None)
w.add_eos_token_id(tid(tc.get("eos_token")) if tc.get("eos_token") and tid(tc.get("eos_token")) is not None else eos)
w.add_bos_token_id(bos if bos is not None else eos); w.add_add_bos_token(False)
if pad and tid(pad) is not None: w.add_pad_token_id(tid(pad))
tpl = tc.get("chat_template")
if not tpl and os.path.exists(os.path.join(a.src, "chat_template.jinja")): tpl = open(os.path.join(a.src, "chat_template.jinja"), encoding="utf-8").read()
if tpl: w.add_chat_template(tpl)

# Gewichte
names = gguf.get_tensor_name_map(gguf.MODEL_ARCH.QWEN2, n_layer)
n = 0
with safe_open(os.path.join(a.src, "model.safetensors"), "pt") as f:
    keys = list(f.keys())
    for k in keys:
        name = names.get_name(k, try_suffixes=(".weight", ".bias"))
        if name is None: sys.exit("Unbekanntes Gewicht: " + k)
        t = f.get_tensor(k).float().numpy()
        if t.ndim == 1 or name.endswith(".bias"): w.add_tensor(name, t.astype(np.float32))
        elif a.typ in ("q8_0", "q4_0") and t.shape[-1] % 32 == 0:
            qt = Q.Q4_0 if (a.typ == "q4_0" and name != "token_embd.weight") else Q.Q8_0  # Wörter-Tabelle bleibt genauer (Q8), der Rest wird kleiner (Q4)
            w.add_tensor(name, gguf.quantize(t.astype(np.float32), qt), raw_dtype=qt)
        else: w.add_tensor(name, t.astype(np.float16))
        n += 1
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
print(f"{n} Gewichte, {os.path.getsize(a.out) // 1048576} MB -> {a.out}")
