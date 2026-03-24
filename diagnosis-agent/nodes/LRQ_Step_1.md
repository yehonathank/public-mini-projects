---
kind: question
patient: "Did the pain start at your belly button and move down, or did it start in the lower right?"
ehr_auto_goto: LRQ_Step_2
ehr_auto_when: appendectomy_in_pmh
---

# Logic:
- **Pre-check:** Before applying **IF EHR** (appendectomy / appendix), call `check_chart` with `category=past_medical_history` unless PMH is already in this turn's tool results.
- IF **EHR** documents prior **appendectomy** / appendix removed (PMH): GOTO node_id **LRQ_Step_2** — skip the classic acute-appendicitis migration branch; pursue alternate LRQ etiologies.
- IF response contains (**belly button** OR **navel** OR **umbilical** OR **moved** OR **migration** OR **migrating**): GOTO node_id **Appendicitis_Result**
- IF response contains (**stayed** OR **started in lower right** OR **always lower right** OR **bottom right** OR **did not start at belly button**): GOTO node_id **LRQ_Step_2**
