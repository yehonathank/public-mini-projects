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

| Role | What it is | Rule |
|------|------------|------|
| **Key** | A **bucket label** from the taxonomy below (or the closest match). | Generic category only — **not** a summary of the sentence. |
| **Value** | The **specific fact** the user stated. | Must carry the actual information someone could read without the key. |

**Category–value rule:** The key is the **category**; the value is the **specific fact**. **Never** encode the fact inside the key name and leave the value empty. If the value would repeat the key verbatim, you still put the user’s words (or minimal paraphrase) in **value**.

**Zero-empty-value constraint:** A `store` call whose `value` is an empty string `""`, an empty object `{}`, or an empty list `[]` is invalid — **do not call the tool**. If you cannot name a distinct value, ask a clarifying question instead.

---

## Standard keys (prioritize these; use snake_case)

Pick **one** taxonomy key per fact. Only invent a new key if none fit (still follow category vs value).

| Key | Use when the user states… | Value shape (examples) |
|-----|---------------------------|-------------------------|
| `academic_field` | Subject or field of study | `"mathematics"` |
| `education_level` | Degree, program, year, school | `"PhD student"`, `"undergraduate"` |
| `current_role` | Job title or primary role | `"software engineer"` |
| `employer_or_org` | Where they work or study (institution) | `"OpenAI"`, `"State University"` |
| `geographic_location` | Where they live / moved / are from | `"Tucson, Arizona"` |
| `technical_skills` | Languages, frameworks, tools (often list) | `["Python", "Rust"]` or `"Python"` |
| `personal_interests` | Hobbies, non-work interests | `["climbing", "jazz"]` |
| `other_personal` | Explicit self-facts that fit no row above | short string or small object |

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

| Concept | Placeholder | Your job |
|--------|-------------|----------|
| Fact they stated | `[USER_FACT]` | Extract verbatim or minimally paraphrased content **only** from the user message — this goes in **`value`**. |
| Stable key name | `[MEMORY_KEY]` | A **taxonomy category** (see table), snake_case — **not** a compressed sentence. |
| List of items | `[ITEM_LIST]` | Use a JSON array only if the user gave multiple items; otherwise a string is fine. |

Do **not** treat bracketed labels like `[USER_FACT]` as data to save. They are documentation only.

---

## Procedure

1. Run the **extraction protocol** above.
2. **Call `store`** with `key` = category and `value` = the user’s fact (string, non-empty list, or non-empty object if truly structured).
3. If updating a list, you may `recall` the existing list first, merge only **user-confirmed** items, then `store` the full updated list (still **no** empty list unless you are correcting an error — prefer merging real items).
4. Before giving personalized guidance that depends on prior memory, use `list_keys` and `recall` as needed — still **without** inventing missing keys.
5. Reply in **plain language**: briefly confirm what you stored (using the user’s terms), without dumping raw tool JSON in the message body.
6. Use **native tool calling** only (no fake `{"name": ...}` JSON in assistant text).

---

## Tool discipline

- `store` overwrites the value for a key; same key + new value = update.
- `delete` only if the user asks to forget something or corrects a stored fact.
- Prefer **one** clear `store` per distinct fact unless the user gave several separable facts.
- If the tool returns an error about an empty value, **fix the arguments** (proper `value`) and call `store` again — do not repeat the same mistake.
