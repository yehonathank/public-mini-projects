---
kind: question
patient: "Did the pain start at your belly button and move down, or did it start in the lower right?"
---

# Logic:
- IF response contains (**belly button** OR **navel** OR **umbilical** OR **moved** OR **migration** OR **migrating**): GOTO node_id **Appendicitis_Result**
- IF response contains (**stayed** OR **started in lower right** OR **always lower right** OR **bottom right** OR **did not start at belly button**): GOTO node_id **LRQ_Step_2**
