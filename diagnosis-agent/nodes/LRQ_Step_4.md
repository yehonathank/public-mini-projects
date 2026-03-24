# State: LRQ_Step_4

# Question: "Is the pain a constant ache, or does it come in waves of intense agony that travel toward your groin?"

# Logic:
- IF response contains (**waves** OR **comes and goes** OR **colicky** OR **groin** OR **radiating to groin** OR **pacing** OR **flank**): GOTO node_id **Kidney_Stone_Result**
- IF response contains (**constant** OR **steady** OR **dull** OR **mid-cycle**): GOTO node_id **LRQ_Step_5**
