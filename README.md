# Pip für Velun

Pip ist der kleine Lern-Begleiter in der App Velun. Er weiß nur, was ihm sein Besitzer beigebracht hat.

- `pip/`: das Programm, das Pips Modell trainiert (LoRA auf Qwen2.5-1.5B-Instruct, Apache-2.0) und in GGUF umwandelt.
- `.github/workflows/pip-update.yml`: läuft jede Nacht in der Cloud. Hat Pip etwas Neues gelernt, wird das Modell neu trainiert und im Zweig `modell` abgelegt. Die App lädt es von dort.
- Zweig `modell`: das fertige Modell in Stücken zu 40 MB.

Zugangsdaten stehen nur in den Secrets dieses Repos, nicht im Code.
