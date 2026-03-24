# State: LRQ_Step_5

# Question: "Does the pain feel like a localized sharp point where you can't feel it anywhere else, or is it a broader pelvic ache?"

# Logic:
- IF response contains (**localized** OR **one spot** OR **sharp point** OR **feel fine otherwise**): GOTO node_id **Appendagitis_Result**
- IF response contains (**pelvic** OR **ovary** OR **ovarian** OR **exercise** OR **broader ache**): GOTO node_id **Ovarian_Cyst_Result**
