#!/usr/bin/env python3
"""Offline checks for GOTO parsing (no API). Run: python test_router_logic.py"""

from ehr_reader import ehr_predicate_holds

from router import (
    allowed_next_node_ids,
    extract_branch_target_node_ids,
    load_node,
    meta_ehr_auto_redirect,
    split_node_document,
)


def main() -> None:
    skills = load_node("skills")
    ts = extract_branch_target_node_ids(skills)
    assert ts == ["LRQ_Step_1", "Other_Complaint"], ts
    assert "skills" in allowed_next_node_ids(skills, "skills")

    s1 = load_node("LRQ_Step_1")
    t1 = extract_branch_target_node_ids(s1)
    assert t1 == ["LRQ_Step_2", "Appendicitis_Result"], t1

    s3 = load_node("LRQ_Step_3")
    assert extract_branch_target_node_ids(s3) == ["Hernia_Result", "Adenitis_Result", "LRQ_Step_4"], (
        extract_branch_target_node_ids(s3)
    )

    oc = load_node("Other_Complaint")
    to = extract_branch_target_node_ids(oc)
    assert to == ["LRQ_Step_1", "Other_Complaint", "Other_Result"], to

    ar = load_node("Appendicitis_Result")
    assert extract_branch_target_node_ids(ar) == []

    s1m, _ = split_node_document(s1)
    g, w = meta_ehr_auto_redirect(s1m)
    assert g == "LRQ_Step_2" and w == "appendectomy_in_pmh"
    ehr_demo = {
        "past_medical_history": ["Appendectomy (2018)"],
        "demographics": {"biological_sex": "Female"},
    }
    assert ehr_predicate_holds("appendectomy_in_pmh", ehr_demo)
    assert not ehr_predicate_holds("biological_sex_male", ehr_demo)
    assert ehr_predicate_holds("biological_sex_male", {"demographics": {"biological_sex": "Male"}})

    print("test_router_logic: ok")


if __name__ == "__main__":
    main()
