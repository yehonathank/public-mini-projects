---
kind: question
patient: "Does the pain feel like a localized sharp point where you can't feel it anywhere else, or is it a broader pelvic ache?"
ehr_auto_goto: Gynecologic_Pruned_Result
ehr_auto_when: biological_sex_male
---

# Logic:
- **Pre-check:** Before applying **IF EHR demographics** (male → gynecologic prune), call `check_chart` with `category=demographics` unless demographics are already in this turn's tool results.
- IF **EHR demographics** indicate **Male** biological sex: GOTO node_id **Gynecologic_Pruned_Result** (ovarian-pathway branch not applicable).
- IF response contains (**localized** OR **one spot** OR **sharp point** OR **feel fine otherwise**): GOTO node_id **Appendagitis_Result**
- IF response contains (**pelvic** OR **ovary** OR **ovarian** OR **exercise** OR **broader ache**): GOTO node_id **Ovarian_Cyst_Result**
