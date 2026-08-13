"""
KaStack L1 - Streamlit demonstration UI.

This file contains NO classification, extraction or detection
logic. Everything is imported from pipeline.py, which is the same
module the notebook verifies, so the UI and the notebook can never
disagree.

Privacy: the supplied assessment dataset is never bundled with
this app. The public demo runs on sample_synthetic.csv, which is
invented data written for demonstration purposes.
"""

import json
import pandas as pd
import streamlit as st

import pipeline as P

st.set_page_config(page_title="KaStack L1 - Message Intelligence",
                   layout="wide")

SAMPLE_PATH = "sample_synthetic.csv"


# ---------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------
@st.cache_data(show_spinner=False)
def run_pipeline(csv_bytes: bytes | None, use_sample: bool):
    """Load a CSV and run the shared pipeline over it."""
    if use_sample:
        df = pd.read_csv(SAMPLE_PATH, dtype=str, keep_default_na=False)
    else:
        from io import BytesIO
        df = pd.read_csv(BytesIO(csv_bytes), dtype=str,
                         keep_default_na=False, encoding="utf-8-sig")
    report = P.validate_dataframe(df)
    if not report["ok"]:
        return None, None, report
    processed, items = P.process_messages(df)
    return processed, items, report


# ---------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------
st.sidebar.title("Data source")
source = st.sidebar.radio(
    "Choose input",
    ["Synthetic demo data", "Upload a CSV"],
    help="The supplied assessment dataset is private and is not "
         "bundled with this app.",
)

uploaded = None
if source == "Upload a CSV":
    uploaded = st.sidebar.file_uploader(
        "CSV with message_id, timestamp, sender, message",
        type=["csv"])
    st.sidebar.warning(
        "Do not upload confidential data to a public deployment. "
        "Run this app locally for private datasets.")

mandatory_ids = []
mand_file = st.sidebar.file_uploader(
    "Optional: mandatory IDs CSV", type=["csv"], key="mand")
if mand_file is not None:
    _m = pd.read_csv(mand_file, dtype=str, encoding="utf-8-sig")
    col = "message_id" if "message_id" in _m.columns else _m.columns[0]
    mandatory_ids = [s.strip() for s in _m[col] if str(s).strip()]

manual_ids = st.sidebar.text_input(
    "Or enter IDs, comma separated", "")
if manual_ids.strip():
    mandatory_ids = [s.strip() for s in manual_ids.split(",") if s.strip()]


# ---------------------------------------------------------------
# Run
# ---------------------------------------------------------------
st.title("Message Intelligence - KaStack L1")
st.caption("Local, deterministic NLP. No external AI or LLM API is "
           "called at any point.")

use_sample = source == "Synthetic demo data"
if not use_sample and uploaded is None:
    st.info("Upload a CSV, or switch to the synthetic demo data.")
    st.stop()

processed, items, report = run_pipeline(
    uploaded.getvalue() if uploaded is not None else None, use_sample)

if processed is None:
    st.error("Validation failed - the file was not processed.")
    for e in report["errors"]:
        st.write(f"- {e}")
    st.stop()

for w in report["warnings"]:
    st.warning(w)

tabs = st.tabs(["Overview", "Classification", "Tasks & Events",
                "Sensitive", "Mandatory IDs", "Approach"])


# --- Overview ---------------------------------------------------
with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Messages", len(processed))
    c2.metric("Tasks & events", len(items))
    c3.metric("Sensitive", int(processed["is_sensitive"].sum()))
    c4.metric("Chronological",
              "yes" if processed["parsed_ts"].is_monotonic_increasing
              else "no")

    st.subheader("Category distribution")
    counts = (processed["category"].value_counts()
              .rename(index=P.CATEGORY_LABELS))
    st.bar_chart(counts)

    st.subheader("Confidence distribution")
    st.caption("Rule-strength scores, not model probabilities. "
               "No classifier was trained and no ground truth exists, "
               "so no accuracy metric is reported.")
    conf = processed["confidence"].value_counts().sort_index()
    st.dataframe(
        pd.DataFrame({
            "confidence": conf.index,
            "messages": conf.values,
            "meaning": [P.CONFIDENCE_MEANING[c] for c in conf.index],
        }), use_container_width=True, hide_index=True)

    st.subheader("Dataset structure")
    st.caption("Sensitive values are masked before display.")
    st.dataframe(
        processed[["message_id", "timestamp", "sender",
                   "masked_text"]].head(20),
        use_container_width=True, hide_index=True)


# --- Classification ---------------------------------------------
with tabs[1]:
    st.subheader("Classification results")
    pick = st.multiselect(
        "Filter by category",
        [P.CATEGORY_LABELS[c] for c in P.CATEGORY_PRECEDENCE],
        default=[P.CATEGORY_LABELS[c] for c in P.CATEGORY_PRECEDENCE])
    keep = [c for c in P.CATEGORY_PRECEDENCE
            if P.CATEGORY_LABELS[c] in pick]
    view = processed[processed["category"].isin(keep)]

    st.caption(f"{len(view)} messages. Every row carries the reason "
               "produced by the rule that made the decision.")
    st.dataframe(
        view[["message_id", "masked_text", "category", "confidence",
              "reason"]],
        use_container_width=True, hide_index=True)


# --- Tasks & events ---------------------------------------------
with tabs[2]:
    st.subheader("Extracted tasks and events")
    st.caption("null = the message never referred to that slot. "
               "'unresolved' = referred to, but no explicit value "
               "given. Nothing is guessed.")

    kind = st.radio("Type", ["all", "task", "event"], horizontal=True)
    iv = items if kind == "all" else items[items["type"] == kind]

    if len(iv):
        st.dataframe(
            iv[["item_id", "type", "title", "deadline", "date_hint",
                "time", "person", "location", "priority",
                "date_in_past", "evidence", "source_message_id"]],
            use_container_width=True, hide_index=True)

        n_past = int((items["date_in_past"] == True).sum())
        if n_past:
            st.info(
                f"{n_past} item(s) carry a deadline earlier than their "
                "own message timestamp. This is a property of the "
                "source data and is preserved, not corrected.")
    else:
        st.write("No items extracted.")


# --- Sensitive ---------------------------------------------------
with tabs[3]:
    st.subheader("Sensitive information")
    sens = processed[processed["is_sensitive"]].copy()
    st.caption("Values are masked at the point of detection. Raw "
               "values are never rendered by this app.")

    if len(sens):
        sens["sensitivity_type"] = sens["sensitivity_types"].map(
            lambda t: ", ".join(t))
        st.dataframe(
            sens[["message_id", "sensitivity_type", "risk_level",
                  "masked_text", "recommended_action"]],
            use_container_width=True, hide_index=True)
    else:
        st.write("No sensitive values detected.")

    ref = processed[processed["is_reference_only"]]
    if len(ref):
        st.subheader("Referenced but not present")
        st.caption("These mention credential concepts but contain no "
                   "actual value, so they are NOT flagged sensitive.")
        st.dataframe(ref[["message_id", "masked_text", "category"]],
                     use_container_width=True, hide_index=True)


# --- Mandatory IDs ------------------------------------------------
with tabs[4]:
    st.subheader("Mandatory demonstration IDs")
    if not mandatory_ids:
        st.info("Upload a mandatory IDs CSV, or enter IDs in the "
                "sidebar, to render them here.")
    else:
        found = [i for i in mandatory_ids
                 if i in set(processed["message_id"])]
        st.write(f"{len(found)} of {len(mandatory_ids)} IDs found.")
        missing = [i for i in mandatory_ids if i not in found]
        if missing:
            st.warning(f"Not in this dataset: {missing}")

        for mid in found:
            r = processed[processed["message_id"] == mid].iloc[0]
            with st.expander(
                    f"{mid} - {P.CATEGORY_LABELS[r['category']]} "
                    f"({r['confidence']:.2f})", expanded=False):
                st.write(f"**Text (masked):** {r['masked_text']}")
                st.write(f"**Sender:** {r['sender']}  |  "
                         f"**Sent:** {r['timestamp']}")
                st.write(f"**Reason:** {r['reason']}")
                if r["is_sensitive"]:
                    st.error(
                        f"Sensitive: {', '.join(r['sensitivity_types'])} "
                        f"| risk {r['risk_level']} | "
                        f"{r['recommended_action']}")
                if r["is_reference_only"]:
                    st.info("References credential information but "
                            "carries no value - not flagged sensitive.")
                sub = items[items["source_message_id"] == mid]
                if len(sub):
                    st.write("**Extracted:**")
                    st.dataframe(sub, use_container_width=True,
                                 hide_index=True)


# --- Approach -----------------------------------------------------
with tabs[5]:
    st.subheader("How it works")
    st.markdown("""
**Pipeline:** validate → minimal preprocessing → sensitive
detection → classification → task/event extraction.

**Classification** walks a precedence ladder: Sensitive →
Promotional → Meeting or Event → Action Required → Personal →
General. Rules match *structural frames* such as
`can you ... before <date>`, not bare keywords, so the filler
prefix "Can you help?" cannot force an Action Required label.

**Extraction** uses named capture groups, so every slot is a real
span of the message. Three states are kept apart: a value,
`unresolved` (referenced but not explicit), and null (never
referenced).

**Sensitive detection** runs on the original text, because
lowercasing destroys the casing that token and ID formats depend
on. Only the captured value is masked.

**Confidence** is rule strength, not probability.
""")
    st.subheader("Limitations")
    st.markdown("""
- Frames are derived from observed message patterns. Unseen
  phrasings fall to General Information at 0.50 rather than being
  forced into a category.
- No ground-truth labels exist for the source corpus, so no
  accuracy, precision, recall or F1 is reported. Doing so would
  require inventing labels.
- Relative expressions ("tomorrow", "next week") are deliberately
  never converted into dates.
- Priority is a documented heuristic derived from deadline
  presence, not a field stated in any message.
""")
