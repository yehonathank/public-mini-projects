#!/usr/bin/env python3
"""Offline checks for GOTO parsing (no API). Run: python test_router_logic.py"""

from router import allowed_next_node_ids, extract_branch_target_node_ids, load_node


def main() -> None:
    skills = load_node("skills")
    ts = extract_branch_target_node_ids(skills)
    assert ts == ["LRQ_Step_1", "Other_Complaint"], ts
    assert "skills" in allowed_next_node_ids(skills, "skills")

    s1 = load_node("LRQ_Step_1")
    t1 = extract_branch_target_node_ids(s1)
    assert t1 == ["Appendicitis_Result", "LRQ_Step_2"], t1

    oc = load_node("Other_Complaint")
    to = extract_branch_target_node_ids(oc)
    assert to == ["LRQ_Step_1", "Other_Complaint", "Other_Result"], to

    ar = load_node("Appendicitis_Result")
    assert extract_branch_target_node_ids(ar) == []

    print("test_router_logic: ok")


if __name__ == "__main__":
    main()
