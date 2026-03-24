# State: skills

# Question: "Is your main concern pain in the lower right part of your abdomen, or something else?"

# Logic:
- IF response affirms **lower right** as the main concern (including bare **yes** when it clearly agrees with that option, or phrases like **lower right**, **RLQ**, **right lower abdomen**, **right side bottom**): GOTO node_id **LRQ_Step_1**
- ELSE (other location, vague without affirming LRQ, or clearly not LRQ): GOTO node_id **Other_Complaint**
