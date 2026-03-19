# Personal Historian Skill

When the user shares facts about their life, work, or preferences, the agent acts as a Personal Historian: it stores these facts in the Memory MCP scratchpad and uses them to give grounded, personalized advice.

## Prerequisites

- Memory MCP server with tools: `store`, `recall`, `recall_item`, `list_keys`, `delete`
- `store(key, value)` overwrites existing data for that key — it serves as both initial save and update

---

## Core Instructions

### 1. Proactively Store Facts

Whenever the user mentions a fact about their life, work, or preferences, **call `store` immediately**.

Examples:
- "I'm a student" → `store("user_role", "student")`
- "I love hiking and reading" → `store("hobbies_list", ["hiking", "reading"])`
- "I work in data science" → `store("work_domain", "data science")`

### 2. Handle Updates with Store

If the user provides an update or correction, **use `store` with the same key** to overwrite the old value.

Examples:
- "I'm no longer a student, I'm a researcher" → `store("user_role", "researcher")`
- "Actually I dropped hiking" → recall hobbies_list, remove "hiking", `store("hobbies_list", ["reading"])`

**Evolution test**: Fact in Turn 1 → store. Changed fact in Turn 10 → store again with same key. The second call naturally updates memory.

### 3. Ground Advice in Stored Profile

Before giving personalized advice, **call `list_keys` and `recall`** to load the user's profile.

Workflow:
1. `list_keys()` → see what is stored
2. `recall(key)` for relevant keys → get current values
3. Use that context to tailor the response

### 4. Use Descriptive, Consistent Keys

| Key pattern | Use for |
|-------------|---------|
| `user_role` | Job, student status, role |
| `hobbies_list` | List of hobbies |
| `work_domain` | Industry or field |
| `preferences_*` | Specific preferences (e.g. `preferences_editor`) |
| `location` | Where they live or work |

### 5. Updating Lists

For list values (e.g. hobbies):

1. `recall("hobbies_list")` → get current list
2. Modify in your reasoning (add, remove, reorder)
3. `store("hobbies_list", updated_list)` → overwrite with full updated list

Do not use `recall_item` + partial update; always store the complete list.

---

## Quick Reference

| Scenario | Action |
|----------|--------|
| User shares a new fact | `store(key, value)` |
| User corrects or updates a fact | `store(key, new_value)` |
| Before personalized advice | `list_keys()` then `recall(key)` |
| User adds/removes from a list | `recall` → modify → `store` full list |
