---
kind: question
patient: "Did the pain start after a specific physical event, like heavy lifting, or have you recently been sick with a cold or sore throat?"
---

# Logic:
- IF response contains (**lifting** OR **heavy** OR **bulge** OR **pop** OR **strained**): GOTO node_id **Hernia_Result**
- IF response contains (**cold** OR **sore throat** OR **virus** OR **flu** OR **recent infection**): GOTO node_id **Adenitis_Result**
- IF response contains (**no** OR **neither** OR **sudden** OR **nothing like that**): GOTO node_id **LRQ_Step_4**
