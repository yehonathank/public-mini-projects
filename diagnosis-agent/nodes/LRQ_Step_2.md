# State: LRQ_Step_2

# Question: "Have you noticed any changes in your bowel habits, like significant constipation, diarrhea, or recent weight loss?"

# Logic:
- IF response contains (**constipation** OR **bloated** OR **previous left side pain** OR **left-sided pain**): GOTO node_id **Diverticulitis_Result**
- IF response contains (**weight loss** OR **chronic diarrhea** OR **long term** OR **ongoing diarrhea**): GOTO node_id **Crohns_Result**
- IF response contains (**no** OR **none** OR **normal** OR **no change**): GOTO node_id **LRQ_Step_3**
