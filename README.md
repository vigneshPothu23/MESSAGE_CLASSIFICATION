# Message Intelligence — KaStack AI/ML Internship (L1)

A local, deterministic NLP system that classifies messages into six
categories, extracts tasks and events, and detects and masks sensitive
information.

**Live demo:** https://messageclassification-ghkq82eboojvmqgsr8r44s.streamlit.app/
**Repository:** https://github.com/vigneshPothu23/MESSAGE_CLASSIFICATION

No external AI or LLM API is called at any point. All processing runs
locally.

---

## Contents

| File | Purpose |
|---|---|
| `Message_Classification.ipynb` | Development notebook: analysis, implementation, validation |
| `pipeline.py` | Shared processing logic — single source of truth |
| `app.py` | Streamlit UI. Imports `pipeline.py`; contains no logic of its own |
| `audit_notebook.py` | Scans notebook outputs for dataset content before committing |
| `sample_synthetic.csv` | Invented demo data for the public app (not the supplied dataset) |
| `requirements.txt` | `pandas`, `streamlit` |

The supplied assessment dataset is **not** in this repository. `data/`
and `outputs/` are gitignored.

---

## Dataset

The supplied dataset contains 900 messages with columns `message_id`,
`timestamp`, `sender`, `message`. Verified properties:

- 900 rows, 0 nulls, 0 duplicate rows, 0 duplicate IDs, 0 duplicate texts
- Timestamps `2026-09-01 08:00:00` → `2026-09-24 10:23:00`, already
  chronological, all unique
- 13 senders; `Promotions` (100 messages) plus 10 individuals and two
  role accounts

Two findings shaped the design:

1. **Sender is not a reliable signal.** `Promotions` sends 100 messages,
   but 110 messages are promotional — 10 come from an individual. A
   sender-based rule would look strong and silently miss 9% of the class.
2. **229 of the 350 dated messages carry a date earlier than their own
   timestamp.** This is a property of the data. The system preserves the
   literal date and exposes `date_in_past: true` rather than correcting it.

---

## Approach

```
validate → minimal preprocessing → sensitive detection →
classification → task/event extraction → structured outputs
```

Deterministic frame matching was chosen over a trained classifier for a
specific reason. The dataset ships with no ground-truth labels. A
TF-IDF + Logistic Regression experiment on rule-derived weak labels
scored **1.0000 under a random split** and **0.8444 ± 0.0544 under a
split grouped by message template**. The perfect score is an artifact:
near-identical templates repeat across the corpus, so a random split puts
the same string in both train and test. Reporting it as accuracy would be
meaningless. The honest number is lower than the rule system's own
agreement, and a trained model would also destroy the `reason` field the
assessment requires. No model is shipped.

### Preprocessing

Only four steps, each justified by the data:

- parse timestamps; verify chronological order (already sorted, so this
  is a verified assertion rather than a re-sort)
- strip 9 known noise prefixes into a separate `core_text` column
- build a lowercased `match_text` used only for case-insensitive matching
- the original `message` column is never modified

Not done, deliberately: stemming, stopword removal, punctuation
stripping, lowercasing the stored text. Case distinguishes `SAVE29`,
`ID-1234-XY`, `RC-88-KL` and `tok_demo_…`, all of which sensitive
detection depends on, and hedging words (`could`, `may`, `might`) drive
the low-confidence band.

---

## 1. Message classification

Six categories, resolved by a precedence ladder — first match wins:

1. **Sensitive Information** — a message carrying a live credential is
   sensitive whatever else it asks
2. **Promotional** — a discount-code frame is decisive
3. **Meeting or Event**
4. **Action Required**
5. **Personal Information**
6. **General Information** — documented fallback, never a silent default

Rules match **structural frames**, not keywords: `can you … before
<date>`, `^calendar update:`, `\bis due on <date>`. This matters because
the filler prefix `Can you help?` appears on 90 messages — 10 of them
discount-code promotions — so a `"can you" → Action Required` rule would
be wrong in both directions.

Every message carries a `reason` emitted by the same rule that made the
decision, so an explanation cannot drift from the logic.

### Confidence

Rule-strength scores, **not** model probabilities:

| Score | Meaning |
|---|---|
| 0.95 | Highly specific pattern — very low ambiguity |
| 0.90 | Clear frame plus a corroborating explicit signal |
| 0.75 | Clear frame, no corroborating detail |
| 0.55 | Hedged or under-specified — genuinely uncertain |
| 0.50 | No frame matched — fallback category |

### Results (900 messages)

| Category | Count |
|---|---|
| Action Required | 240 |
| Meeting or Event | 170 |
| General Information | 170 |
| Personal Information | 110 |
| Promotional | 110 |
| Sensitive Information | 100 |

Confidence spread: 450 at 0.95, 200 at 0.90, 20 at 0.75, 60 at 0.55,
170 at 0.50. Roughly 26% of the corpus is explicitly marked uncertain
rather than presented as confident.

---

## 2. Task and event extraction

16 frames with named capture groups, so every extracted slot is a real
span of the message. Three states are kept distinct:

- a **value** — explicitly present
- `"unresolved"` — referenced, but no explicit value given
  (*"sometime next week"*, *"Friday afternoon"*)
- `null` — never referenced at all

That distinction is the core of the "do not guess" requirement. A message
reading *"Please call Maya when you are free"* yields deadline `null`,
not `unresolved`, because no timing was ever mentioned.

Relative expressions are **never** converted into dates. `tomorrow` stays
`unresolved` with the surface phrase preserved in `date_hint`.

`priority` is a documented heuristic derived from deadline presence
(`high` with an explicit date, `medium` for an undated imperative, `low`
when hedged). It is not a field stated in any message.

### Results

410 items: **240 tasks, 170 events**. Deadlines: 350 explicit, 50
unresolved, 10 not referenced. 229 flagged `date_in_past`.

---

## 3. Sensitive information

Ten labelled patterns run against the **original** message text, since
lowercasing would destroy token and ID casing. Each pattern captures only
the value, so masking hides the secret while leaving the sentence
readable.

| Type | Risk | Recommended action |
|---|---|---|
| one_time_password, password, auth_token, recovery_code, id_number | high | `do_not_store` |
| bank_account, card_number | high | `do_not_send_to_external_service` |
| health_data | high | `ask_for_confirmation` |
| home_address, phone_number | medium | `ask_for_confirmation` |

**Value present vs. reference.** *"I will send the login details
separately"* names a credential concept but contains no value. Those 10
messages are tracked as `is_reference_only` and are **not** classified
Sensitive.

### Results

100 sensitive messages, exactly 10 per type — 80 high risk, 20 medium.
10 reference-only.

---

## 4. Validation

21 automated checks, all passing:

- coverage — 900 processed, no IDs lost, all unique
- chronological processing verified
- all six categories present; every message has a category and a reason;
  every confidence within the declared bands
- fallback confidence appears only on General Information
- every item has a valid `source_message_id`; no item extracted from a
  sensitive message
- **every explicit deadline is a literal ISO date and was verified to
  appear in its own source message** — mechanical proof no date was
  invented; the same check runs for extracted times and named persons
- all 15 mandatory IDs processed
- zero raw sensitive values in any artifact

**Leakage test.** Raw sensitive values are harvested from the source
messages and every output file is scanned for them — including a re-scan
reading the written files back off disk. The scanner is self-tested
against a deliberately poisoned sample, because a leakage test that
cannot fail proves nothing. Result: **0 leaks**.

**No hardcoding.** `inspect.getsource` over the six pipeline functions
confirms zero mandatory-ID literals. Each rule that fires on the
mandatory 15 also fires across the wider corpus.

**Determinism.** Two runs produce identical output. `pipeline.py` was
verified to reproduce the notebook results across 10 compared fields.

No accuracy, precision, recall or F1 is reported. No ground-truth labels
exist, and producing such a metric would require inventing them.

---

## Running it

```bash
pip install -r requirements.txt

# Streamlit UI
streamlit run app.py

# Notebook
jupyter notebook Message_Classification.ipynb
```

The notebook expects `data/messages.csv` and
`data/mandatory_demo_ids.csv` locally. Neither is distributed here.

The public app defaults to `sample_synthetic.csv`. A CSV uploader is
provided for local use with private data — **do not upload confidential
data to the public deployment.**

---

## Assumptions

- The 9 noise prefixes are conversational filler carrying no category
  signal
- `Please note:` (with colon) is a prefix; `Please note my bank account
  number …` is content — the colon is the only separator
- Precedence ordering reflects that credential disclosure outranks any
  other reading of a message
- Priority is inferred from deadline presence, not stated in messages
- Dates earlier than their message timestamp are dataset properties, not
  errors to repair

## Limitations

- Frames are derived from observed patterns. An unseen phrasing falls to
  General Information at 0.50 rather than being forced into a category —
  the honest failure mode, but it means novel corpora need new frames.
- 170 messages (19%) land in the fallback. Most are genuinely plain
  statements, but the fallback cannot distinguish "no frame applies" from
  "a frame exists that I did not write."
- Ambiguous cases are decided by documented convention, not ground truth.
  *"Maya asked whether the demo was ready"* is classified General
  Information because it contains no imperative directed at the reader,
  but a reasonable evaluator could call it Action Required.
- Sensitive detection is pattern-based and requires a labelled context
  word. A credential presented without one would be missed.
- `sensitivity_type` supports multiple types per message, though no
  message in this corpus triggers more than one.
- Extraction handles one item per message; a message describing two
  meetings would yield one.

## AI tool usage declaration

Claude (Anthropic) was used as a development assistant throughout this
assessment. Specifically, it assisted with:

- exploratory analysis of the supplied dataset
- proposing the architecture and the deterministic-over-ML decision
- authoring the regex frames, pipeline code, Streamlit UI and tests
- the TF-IDF leakage experiment that informed the no-ML decision

The dataset was uploaded to Claude during the analysis and development
phase. All design decisions were reviewed and approved by me before
implementation, every cell was executed and verified locally on my own
machine, and the outputs reported here are from those actual runs.

**The submitted application calls no external AI or LLM API.** All
runtime processing is local, using `pandas` and Python's standard
library. No message is transmitted anywhere by the running system.
