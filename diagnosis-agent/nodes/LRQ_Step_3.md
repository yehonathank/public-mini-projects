---
kind: question
patient: "Did the pain start after a specific physical event, like heavy lifting, or have you recently been sick with a cold or sore throat?"
---

# Logic:
- IF response contains (**lifting** OR **heavy** OR **bulge** OR **pop** OR **strained**): GOTO node_id **Hernia_Result**
- IF response contains (**cold** OR **sore throat** OR **virus** OR **flu** OR **recent infection**): GOTO node_id **Adenitis_Result**
- **Pre-check:** Before applying the **IF EHR** `recent_encounters` branch below, call `check_chart` with `category=recent_encounters` unless that slice is already in this turn's tool results.
- IF **EHR** `recent_encounters` documents **sore throat**, **pharyngitis**, **URI**, or similar acute infection AND the patient affirms (**cold** OR **sore throat** OR **virus** OR **flu** OR **sick** OR **cough**): GOTO node_id **Adenitis_Result** — chart aligns with a recent infection visit; avoid redundant questioning when the patient confirms viral symptoms.
- IF response describes (**fall** OR **fell** OR **ladder** OR **accident** OR **hit** OR **struck** OR **impact** OR **trauma** OR **injured** OR **blow** OR **crash** OR **slip** OR **trip**): GOTO node_id **LRQ_Step_4** — mechanical / other event not captured by lifting vs viral dichotomy; continue LRQ workup.
- IF response contains (**no** OR **neither** OR **sudden** OR **nothing like that**): GOTO node_id **LRQ_Step_4**
