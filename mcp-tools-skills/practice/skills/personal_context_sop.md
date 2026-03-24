# Personal Context Manager — Standard Operating Procedure

This document loads **only after** routing has matched the user message to this skill. Until then, you must not follow these steps.

---

## Zero-inference rule (mandatory)

- **Only** call `store` when the user has provided a **direct, explicit statement of fact** about themselves in the current turn (or clearly affirmed a fact you are mirroring back).
- **Never** assume, guess, or invent keys or values from this document, from examples, or from typical user personas.
- **Never** populate `store` using placeholder text or imagined details. Use only what the user actually said.
- If you are unsure whether something is a user-stated fact, **do not** call `store`; answer conversationally or ask a short clarifying question.

---

## Taxonomic extraction (mandatory before every `store`)

Models often confuse **category** (key) with **content** (value). Treat them as **different slots**:


| Role      | What it is                                                         | Rule                                                                  |
| --------- | ------------------------------------------------------------------ | --------------------------------------------------------------------- |
| **Key**   | A **bucket label** from the taxonomy below (or the closest match). | Generic category only — **not** a summary of the sentence.            |
| **Value** | The **specific fact** the user stated.                             | Must carry the actual information someone could read without the key. |


**Category–value rule:** The key is the **category**; the value is the **specific fact**. **Never** encode the fact inside the key name and leave the value empty. If the value would repeat the key verbatim, you still put the user’s words (or minimal paraphrase) in **value**.

**Zero-empty-value constraint:** A `store` call whose `value` is an empty string `""`, an empty object `{}`, or an empty list `[]` is invalid — **do not call the tool**. If you cannot name a distinct value, ask a clarifying question instead.

---

## Standard keys (prioritize these; use snake_case)

Pick **one** taxonomy key per fact. Only invent a new key if none fit (still follow category vs value).


| Key                   | Use when the user states…                 | Value shape (examples)             |
| --------------------- | ----------------------------------------- | ---------------------------------- |
| `academic_field`      | Subject or field of study                 | `"mathematics"`                    |
| `education_level`     | Degree, program, year, school             | `"PhD student"`, `"undergraduate"` |
| `current_role`        | Job title or primary role                 | `"software engineer"`              |
| `employer_or_org`     | Where they work or study (institution)    | `"OpenAI"`, `"State University"`   |
| `geographic_location` | Where they live / moved / are from        | `"Tucson, Arizona"`                |
| `technical_skills`    | Languages, frameworks, tools (often list) | `["Python", "Rust"]` or `"Python"` |
| `personal_interests`  | Hobbies, non-work interests               | `["climbing", "jazz"]`             |
| `other_personal`      | Explicit self-facts that fit no row above | short string or small object       |


---

## Mandatory extraction protocol (do this before calling `store`)

1. **Quote** (mentally) the shortest phrase from the user that is the fact.
2. **Category:** Choose the **taxonomy key** that best fits (e.g. *I study mathematics* → category `academic_field`).
3. **Value:** Put **only the fact** in `value` (e.g. `"mathematics"`). The key must **not** contain *mathematics* as the only place that information lives.
4. **Sanity check:** If someone reads `value` alone, do they learn what the user said? If not, fix `value` — not the key.
5. **Empty check:** If `value` would be `""`, `{}`, or `[]`, **stop** — do not call `store`.

---

## Negative examples (do **not** do this)

Wrong — fact baked into key, value useless:

- `store(key="studies_mathematics", value="{}")`
- `store(key="is_from_tucson", value="")`
- `store(key="user_likes_python_and_rust", value={})`

Right — category in key, fact in value:

- `store(key="academic_field", value="mathematics")`
- `store(key="geographic_location", value="Tucson")`
- `store(key="technical_skills", value=["Python", "Rust"])`

Wrong — key is a full sentence:

- `store(key="i_work_as_a_teacher", value="yes")`

Right:

- `store(key="current_role", value="teacher")`

---

## Variable placeholders (use mentally; do not store placeholders)

When recording memory, map the user’s actual words to structured keys:


| Concept          | Placeholder    | Your job                                                                                                     |
| ---------------- | -------------- | ------------------------------------------------------------------------------------------------------------ |
| Fact they stated | `[USER_FACT]`  | Extract verbatim or minimally paraphrased content **only** from the user message — this goes in `**value`**. |
| Stable key name  | `[MEMORY_KEY]` | A **taxonomy category** (see table), snake_case — **not** a compressed sentence.                             |
| List of items    | `[ITEM_LIST]`  | Use a JSON array only if the user gave multiple items; otherwise a string is fine.                           |


Do **not** treat bracketed labels like `[USER_FACT]` as data to save. They are documentation only.

---

## When the user adds, edits, or updates (mandatory — read before `store`)

Imperatives and deltas — *add …*, *change …*, *update …*, *now it’s …*, *no longer …*, *replace …* — mean the user is changing **their saved profile**, not chatting abstractly. For those turns you **must** use tools; a reply that only talks in natural language and never calls `store` (or `delete` when they ask to forget) is wrong.

**Workflow (always in this order):**

1. **Recall what is already there.** If you are not sure which key holds the old fact, call `list_keys`, then `recall` (or `recall_item` when the value is list-like) on the **most likely** taxonomy key(s) — e.g. `personal_interests` / `hobbies` for “add swimming to my hobbies”, color or `other_personal` for “change my favorite color to blue”, `geographic_location` for “I moved to … now”, `other_personal` or project-related keys for renames.
2. **If something is already stored** under a matching key: build the **new** value from (a) what `recall` returned and (b) the user’s instruction — e.g. **append** to a list for “add X”, **replace** a string for “change to Y”, **rewrite** text for corrections. Then call `**store`** with the **same** key and the new value. In this memory backend, `**store` overwrites** the previous value for that key; that **is** how an update works — you are not “patching” in place without `store`.
3. **If nothing is there** (empty recall, "not found", or no sensible key yet): treat the turn as **new** information — run the **extraction protocol**, pick the right taxonomy **key**, and `**store`** once with a non-empty **value** drawn only from what the user said. **You are not done with the turn** until that `store` has run and returned — a `recall` miss is a signal to **write next**, not to answer the user as if memory were already updated.
4. **Only use `delete`** when the user explicitly wants something **removed** or forgotten, or when removing the key is clearer than overwriting (e.g. fully retracting a stored location). Otherwise prefer **recall → merged/replaced value → `store`**.

Short rule: **look up first, then write.** Skipping `recall` on update/add turns leads to duplicate keys, lost list items, or no write at all. **Ending with only `recall` when the user gave a storable fact and the key was missing is wrong** — call `store` before your final reply.

---

## Procedure

1. If the turn is an **add / edit / update** (see section above), follow that **recall-then-store** workflow **before** anything else.
2. Otherwise run the **extraction protocol** above.
3. **Call `store`** with `key` = category and `value` = the user’s fact (string, non-empty list, or non-empty object if truly structured).
4. For **list updates** after `recall`: merge only **user-confirmed** items into the full list, then `store` the complete new list (still **no** empty list unless correcting an error).
5. Before giving personalized guidance that depends on prior memory, use `list_keys` and `recall` as needed — still **without** inventing missing keys.
6. Reply in **plain language**: briefly confirm what you stored (using the user’s terms), without dumping raw tool JSON in the message body.
7. Use **native tool calling** only (no fake `{"name": ...}` JSON in assistant text).

---

## Tool discipline

- **Truth in what you tell the user:** Do **not** say you stored, saved, or updated memory unless a `**store`** call in **this same turn** already completed successfully **before** your final natural-language message. If you only ran `recall` and the key was missing, your next step is `**store`**, not a verbal claim like 'I've stored that'.
- `**recall` / `list_keys` before `store` on updates:** If the user is adding to, changing, or replacing something they may already have saved, **recall first**, then `**store`** the new whole value. Same key + new value = replacement; that is the supported update path.
- `store` overwrites the value for a key; there is no separate “merge” API — you must read the old value, compute the new value yourself, then `store`.
- `delete` only if the user asks to forget something, or a delete is the right way to retract a fact (see add/edit/update section).
- Prefer **one** clear `store` per distinct fact unless the user gave several separable facts (or you legitimately need multiple keys).
- If the tool returns an error about an empty value, **fix the arguments** (proper `value`) and call `store` again — do not repeat the same mistake.

