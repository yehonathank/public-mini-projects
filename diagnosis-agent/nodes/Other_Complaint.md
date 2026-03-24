# State: Other_Complaint

# Question: "This scripted path only covers lower-right abdominal pain. In a few words, what is your main concern?"

# Logic:
- IF patient clarifies **lower right** / **RLQ** / **right lower abdomen** as the main issue: GOTO node_id **LRQ_Step_1**
- IF response is unusably vague: GOTO node_id **Other_Complaint** (same node — re-ask once if needed)
- ELSE: GOTO node_id **Other_Result**
