# Legacy note

Personal-context behavior is now driven by:

- **`skill_manifest.yaml`** — routing metadata only (what triggers the skill).
- **`skills/personal_context_sop.md`** — full procedure (loaded **after** a YES route).

The **`ollama_host.py`** agent uses that pipeline by default. This file is kept for human reference only; the host does **not** load it automatically.
