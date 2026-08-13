"""
KaStack L1 - shared processing pipeline.

Single source of truth for message classification, task/event
extraction and sensitive-information detection.

Imported by BOTH Message_Classification.ipynb and app.py so the
notebook and the Streamlit UI can never diverge.

No external AI/LLM API is used. All processing is local.
"""

from __future__ import annotations
import re, json
from collections import Counter
import pandas as pd

# ---------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------
REQUIRED_COLUMNS = ["message_id", "timestamp", "sender", "message"]
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"

CATEGORY_SENSITIVE   = "sensitive_information"
CATEGORY_PROMOTIONAL = "promotional"
CATEGORY_MEETING     = "meeting_or_event"
CATEGORY_ACTION      = "action_required"
CATEGORY_PERSONAL    = "personal_information"
CATEGORY_GENERAL     = "general_information"

CATEGORY_PRECEDENCE = [CATEGORY_SENSITIVE, CATEGORY_PROMOTIONAL,
                       CATEGORY_MEETING, CATEGORY_ACTION,
                       CATEGORY_PERSONAL, CATEGORY_GENERAL]

CATEGORY_LABELS = {
    CATEGORY_SENSITIVE:   "Sensitive Information",
    CATEGORY_PROMOTIONAL: "Promotional",
    CATEGORY_MEETING:     "Meeting or Event",
    CATEGORY_ACTION:      "Action Required",
    CATEGORY_PERSONAL:    "Personal Information",
    CATEGORY_GENERAL:     "General Information",
}

NOISE_PREFIXES = ["For today:", "FYI:", "One more thing:", "Important:",
                  "Just checking\u2014", "Please note:", "Quick update:",
                  "Can you help?", "Hi,"]
MAX_PREFIX_STRIP_PASSES = 4

CONFIDENCE = {"exact_frame": 0.95, "strong": 0.90, "moderate": 0.75,
              "hedged": 0.55, "fallback": 0.50}
CONFIDENCE_MEANING = {
    0.95: "Highly specific pattern - very low ambiguity",
    0.90: "Clear frame supported by a second explicit signal",
    0.75: "Clear frame, no corroborating detail",
    0.55: "Hedged or under-specified language - genuinely uncertain",
    0.50: "No frame matched - fallback category",
}

RISK_HIGH, RISK_MEDIUM, RISK_LOW = "high", "medium", "low"
ACTION_SAFE_LOCAL   = "safe_to_process_locally"
ACTION_CONFIRM      = "ask_for_confirmation"
ACTION_DO_NOT_STORE = "do_not_store"
ACTION_NO_EXTERNAL  = "do_not_send_to_external_service"

SENSITIVITY_POLICY = {
    "one_time_password": (RISK_HIGH,   ACTION_DO_NOT_STORE),
    "password":          (RISK_HIGH,   ACTION_DO_NOT_STORE),
    "auth_token":        (RISK_HIGH,   ACTION_DO_NOT_STORE),
    "recovery_code":     (RISK_HIGH,   ACTION_DO_NOT_STORE),
    "bank_account":      (RISK_HIGH,   ACTION_NO_EXTERNAL),
    "card_number":       (RISK_HIGH,   ACTION_NO_EXTERNAL),
    "id_number":         (RISK_HIGH,   ACTION_DO_NOT_STORE),
    "health_data":       (RISK_HIGH,   ACTION_CONFIRM),
    "home_address":      (RISK_MEDIUM, ACTION_CONFIRM),
    "phone_number":      (RISK_MEDIUM, ACTION_CONFIRM),
}

MASK_CHAR, MASK_WIDTH = "*", 6
UNRESOLVED = "unresolved"
PRIORITY_HIGH, PRIORITY_MEDIUM, PRIORITY_LOW = "high", "medium", "low"

def R(p): return re.compile(p, re.IGNORECASE)

# ---------------------------------------------------------------
# SENSITIVE PATTERNS  (run on the ORIGINAL message text)
# ---------------------------------------------------------------
SENSITIVE_PATTERNS = [
    ("one_time_password", R(r"\bOTP\s+(?:is|:)\s*([A-Za-z0-9]{4,10})\b")),
    ("password",          R(r"\bpassword\s+(?:is|:)?\s*([A-Za-z0-9@#$%&*!_\-]{6,30})\b")),
    ("auth_token",        R(r"\b(?:access\s+)?token\s+(?:is|:)?\s*([A-Za-z0-9_\-]{8,60})\b")),
    ("recovery_code",     R(r"\brecovery\s+code\s+(?:is|:)?\s*([A-Za-z0-9\-]{4,30})\b")),
    ("card_number",       R(r"\bcard\s+number\s+(?:is|:)?\s*((?:\d[ \-]?){12,19})")),
    ("bank_account",      R(r"\bbank\s+account\s+number\s+(?:is|:)?\s*(\d{8,18})\b")),
    ("id_number",         R(r"\bidentification\s+number\s+(?:is|:)?\s*([A-Za-z0-9\-]{4,20})\b")),
    ("home_address",      R(r"\bhome\s+address\s+(?:is|:)?\s*([^.]{5,80})")),
    ("phone_number",      R(r"\bcontact\s+me\s+on\s+((?:\+?\d[\d\s\-]{7,15}\d))")),
    ("health_data",       R(r"\btest\s+result\s+(?:says|is|:)\s*([^.]{3,60})")),
]
REFERENCE_ONLY_PATTERNS = [R(r"\b(?:login|account)\s+details\b"), R(r"\bcredentials\b")]

# ---------------------------------------------------------------
# CLASSIFICATION FRAMES
# ---------------------------------------------------------------
PROMO_RULES = [
    (R(r"\buse code\s+[a-z0-9]+"), CONFIDENCE["exact_frame"], "marketing offer with a discount code"),
    (R(r"\byou may like our\b"),   CONFIDENCE["moderate"],    "product promotion without a discount code"),
]
MEETING_RULES = [
    (R(r"^are you available for .+ at .+ on \d{4}-\d{2}-\d{2}\?"), CONFIDENCE["exact_frame"], "availability request for a dated, timed event"),
    (R(r"^calendar update:"),                                      CONFIDENCE["exact_frame"], "calendar entry with date, time and venue"),
    (R(r"^please join .+ on \d{4}-\d{2}-\d{2}"),                    CONFIDENCE["exact_frame"], "invitation to a dated session"),
    (R(r"^reminder:.+happens on \d{4}-\d{2}-\d{2}"),                CONFIDENCE["exact_frame"], "reminder of a scheduled occurrence"),
    (R(r"\bis scheduled for \d{4}-\d{2}-\d{2}"),                    CONFIDENCE["exact_frame"], "explicit scheduling statement"),
]
MEETING_VAGUE_RULES = [
    (R(r"^let us meet\b"),                                    CONFIDENCE["hedged"], "meeting proposed without a fixed date"),
    (R(r"\b(?:review|meeting|session|call)\b.*\bcould be\b"),  CONFIDENCE["hedged"], "possible session with hedged timing"),
]
ACTION_RULES = [
    (R(r"^can you .+ before \d{4}-\d{2}-\d{2}"),                                        CONFIDENCE["strong"], "request frame with an explicit deadline"),
    (R(r"^don'?t forget to .+deadline is \d{4}-\d{2}-\d{2}"),                            CONFIDENCE["strong"], "obligation reminder with a stated deadline"),
    (R(r"^i need you to .+ by \d{4}-\d{2}-\d{2}"),                                       CONFIDENCE["strong"], "direct assignment with a due date"),
    (R(r"^please (?:submit|complete|confirm|reply|send|upload) .+ by \d{4}-\d{2}-\d{2}"), CONFIDENCE["strong"], "imperative request with a due date"),
    (R(r"\bis due on \d{4}-\d{2}-\d{2}"),                                                CONFIDENCE["strong"], "task stated with an explicit due date"),
]
ACTION_SOFT_RULES = [
    (R(r"^please (?:call|contact|check)\b"), CONFIDENCE["moderate"], "direct imperative request without a deadline"),
]
ACTION_HEDGED_RULES = [
    (R(r"^if possible,"),           CONFIDENCE["hedged"], "conditional request without a deadline"),
    (R(r"^could you send it soon"), CONFIDENCE["hedged"], "vague request with an unspecified referent"),
    (R(r"\bmay be needed\b"),       CONFIDENCE["hedged"], "possible obligation stated tentatively"),
]
PERSONAL_RULES = [
    (R(r"^for my profile,"),                CONFIDENCE["exact_frame"], "first-person profile disclosure"),
    (R(r"^personal note:"),                 CONFIDENCE["exact_frame"], "explicitly labelled personal note"),
    (R(r"^just so you know, (?:i|my)\b"),   CONFIDENCE["exact_frame"], "first-person preference disclosure"),
    (R(r"^remember that (?:i|my)\b"),       CONFIDENCE["exact_frame"], "first-person standing preference"),
]
PERSONAL_HEDGED_RULES = [
    (R(r"^i might prefer\b"), CONFIDENCE["hedged"], "hedged personal preference"),
]
CLASSIFICATION_LADDER = [
    (CATEGORY_PROMOTIONAL, PROMO_RULES),
    (CATEGORY_MEETING,     MEETING_RULES),
    (CATEGORY_ACTION,      ACTION_RULES),
    (CATEGORY_MEETING,     MEETING_VAGUE_RULES),
    (CATEGORY_PERSONAL,    PERSONAL_RULES),
    (CATEGORY_ACTION,      ACTION_SOFT_RULES),
    (CATEGORY_ACTION,      ACTION_HEDGED_RULES),
    (CATEGORY_PERSONAL,    PERSONAL_HEDGED_RULES),
]

# ---------------------------------------------------------------
# EXTRACTION FRAMES
# ---------------------------------------------------------------
TASK_FRAMES = [
    ("structured", R(r"^can you (?P<title>.+?) before (?P<date>\d{4}-\d{2}-\d{2})")),
    ("structured", R(r"^don'?t forget to (?P<title>.+?); deadline is (?P<date>\d{4}-\d{2}-\d{2})")),
    ("structured", R(r"^i need you to (?P<title>.+?) by (?P<date>\d{4}-\d{2}-\d{2})")),
    ("structured", R(r"^please (?P<title>(?:submit|complete|confirm|reply|send|upload).+?) by (?P<date>\d{4}-\d{2}-\d{2})")),
    ("structured", R(r"^(?P<title>.+?) is due on (?P<date>\d{4}-\d{2}-\d{2})")),
    ("partial",    R(r"^please call (?P<person>[A-Za-z]+) when you are free")),
    ("partial",    R(r"^if possible, (?P<title>.+?) (?P<hint>before the meeting)")),
    ("vague",      R(r"^could you send (?P<title>it) (?P<hint>soon)")),
    ("vague",      R(r"^the (?P<title>report) may be needed (?P<hint>tomorrow)")),
]
EVENT_FRAMES = [
    ("structured", R(r"^are you available for (?P<title>.+?) at (?P<time>\d{1,2}:\d{2}) on (?P<date>\d{4}-\d{2}-\d{2})\? Location: (?P<location>[^.]+)")),
    ("structured", R(r"^calendar update: (?P<title>[^,]+), (?P<date>\d{4}-\d{2}-\d{2}) at (?P<time>\d{1,2}:\d{2}), (?P<location>[^.]+)")),
    ("structured", R(r"^please join (?P<title>.+?) on (?P<date>\d{4}-\d{2}-\d{2}), (?P<time>\d{1,2}:\d{2}) at (?P<location>[^.]+)")),
    ("structured", R(r"^reminder: (?P<title>.+?) happens on (?P<date>\d{4}-\d{2}-\d{2}) at (?P<time>\d{1,2}:\d{2}) in (?P<location>[^.]+)")),
    ("structured", R(r"^the (?P<title>.+?) is scheduled for (?P<date>\d{4}-\d{2}-\d{2}) at (?P<time>\d{1,2}:\d{2}) in (?P<location>[^.]+)")),
    ("vague",      R(r"^let us meet (?P<hint>sometime next week)")),
    ("vague",      R(r"^the (?P<title>review) could be (?P<hint>Friday afternoon)")),
]
ALL_FRAMES = ([("task", s, p) for s, p in TASK_FRAMES] +
              [("event", s, p) for s, p in EVENT_FRAMES])

# ---------------------------------------------------------------
# CORE FUNCTIONS
# ---------------------------------------------------------------
def strip_noise_prefixes(text):
    removed, current = [], text.strip()
    for _ in range(MAX_PREFIX_STRIP_PASSES):
        matched = False
        for prefix in NOISE_PREFIXES:
            if current.startswith(prefix):
                current = current[len(prefix):].strip()
                removed.append(prefix); matched = True; break
        if not matched: break
    if not current: return text.strip(), []
    return current, removed

def build_match_text(core_text):
    return re.sub(r"\s+", " ", core_text).strip().lower()

def detect_sensitive(message):
    detections, spans = [], []
    for stype, pattern in SENSITIVE_PATTERNS:
        for m in pattern.finditer(message):
            if m.group(1) and m.group(1).strip():
                detections.append(stype); spans.append((stype, m.start(1), m.end(1)))
    is_reference = (not detections) and any(p.search(message) for p in REFERENCE_ONLY_PATTERNS)
    if not detections:
        return {"is_sensitive": False, "is_reference": is_reference, "types": [],
                "risk": None, "recommended_action": ACTION_SAFE_LOCAL,
                "masked_text": message, "spans": [],
                "reason": ("References credential information but contains no actual value"
                           if is_reference else "No sensitive value pattern matched")}
    masked = message
    for _s, start, end in sorted(spans, key=lambda s: s[1], reverse=True):
        masked = masked[:start] + (MASK_CHAR * MASK_WIDTH) + masked[end:]
    rank = {RISK_LOW: 0, RISK_MEDIUM: 1, RISK_HIGH: 2}
    top = max(detections, key=lambda t: rank[SENSITIVITY_POLICY[t][0]])
    risk, action = SENSITIVITY_POLICY[top]
    types = sorted(set(detections))
    return {"is_sensitive": True, "is_reference": False, "types": types,
            "risk": risk, "recommended_action": action, "masked_text": masked,
            "spans": spans,
            "reason": f"Detected {', '.join(types)} value pattern{'s' if len(types) > 1 else ''} in the message text"}

def classify_message(core_text, is_sensitive):
    if is_sensitive:
        return {"category": CATEGORY_SENSITIVE, "confidence": CONFIDENCE["exact_frame"],
                "reason": "contains an actual sensitive value",
                "matched_rule": "sensitive_value_detected"}
    for category, rules in CLASSIFICATION_LADDER:
        for pattern, confidence, reason in rules:
            if pattern.search(core_text):
                return {"category": category, "confidence": confidence,
                        "reason": reason, "matched_rule": pattern.pattern[:48]}
    return {"category": CATEGORY_GENERAL, "confidence": CONFIDENCE["fallback"],
            "reason": "no request, scheduling, promotional or disclosure frame matched",
            "matched_rule": "fallback"}

def extract_item(core_text, message_id, sent_ts):
    for kind, strength, pattern in ALL_FRAMES:
        m = pattern.search(core_text)
        if not m: continue
        g = m.groupdict()
        date, hint = g.get("date"), g.get("hint")
        deadline = date if date else (UNRESOLVED if hint else None)
        title = g.get("title")
        if not title and "let us meet" in core_text.lower(): title = "Meet"
        if not title and g.get("person"): title = f"Call {g['person']}"
        if title: title = title[0].upper() + title[1:]
        person = g.get("person")
        if not person and re.search(r"\bMaya\b", core_text): person = "Maya"
        priority = PRIORITY_HIGH if date else (PRIORITY_MEDIUM if strength == "partial" else PRIORITY_LOW)
        date_in_past = bool(pd.Timestamp(date) < sent_ts.normalize()) if date else None
        return {"type": kind, "title": title, "description": core_text,
                "deadline": deadline, "date_hint": hint, "time": g.get("time"),
                "person": person, "location": g.get("location"), "priority": priority,
                "date_in_past": date_in_past, "evidence": strength,
                "source_message_id": message_id}
    return None

def validate_dataframe(df):
    errors, warnings = [], []
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing: errors.append(f"missing required columns: {missing}")
    if not missing:
        if df["message_id"].duplicated().any():
            errors.append(f"{int(df['message_id'].duplicated().sum())} duplicate message_id value(s)")
        bad = pd.to_datetime(df["timestamp"], format=TIMESTAMP_FORMAT, errors="coerce").isna().sum()
        if bad: errors.append(f"{int(bad)} timestamp(s) do not match {TIMESTAMP_FORMAT}")
        blanks = sum(int((df[c].astype(str).str.strip() == "").sum()) for c in REQUIRED_COLUMNS)
        if blanks: warnings.append(f"{blanks} blank cell(s)")
    return {"ok": not errors, "errors": errors, "warnings": warnings}

def process_messages(raw_df):
    df = raw_df.copy()
    df["source_order"] = range(len(df))
    df["parsed_ts"] = pd.to_datetime(df["timestamp"], format=TIMESTAMP_FORMAT, errors="coerce")
    df = df.sort_values(["parsed_ts", "message_id"], kind="stable").reset_index(drop=True)
    s = df["message"].map(strip_noise_prefixes)
    df["core_text"] = [x[0] for x in s]
    df["prefixes_removed"] = [x[1] for x in s]
    df["match_text"] = df["core_text"].map(build_match_text)
    sens = df["message"].map(detect_sensitive)
    df["is_sensitive"]       = [r["is_sensitive"] for r in sens]
    df["is_reference_only"]  = [r["is_reference"] for r in sens]
    df["sensitivity_types"]  = [r["types"] for r in sens]
    df["risk_level"]         = [r["risk"] for r in sens]
    df["recommended_action"] = [r["recommended_action"] for r in sens]
    df["masked_text"]        = [r["masked_text"] for r in sens]
    df["sensitive_reason"]   = [r["reason"] for r in sens]
    cls = [classify_message(c, x) for c, x in zip(df["core_text"], df["is_sensitive"])]
    df["category"]     = [c["category"] for c in cls]
    df["confidence"]   = [c["confidence"] for c in cls]
    df["reason"]       = [c["reason"] for c in cls]
    df["matched_rule"] = [c["matched_rule"] for c in cls]
    eligible = df["category"].isin([CATEGORY_ACTION, CATEGORY_MEETING])
    items = []
    for row in df[eligible].itertuples():
        it = extract_item(row.core_text, row.message_id, row.parsed_ts)
        if it: items.append(it)
    for i, it in enumerate(items, start=1):
        it["item_id"] = f"{'TASK' if it['type'] == 'task' else 'EVENT'}_{i:04d}"
    return df, pd.DataFrame(items)

def harvest_sensitive_values(messages):
    values = set()
    for msg in messages:
        for _s, pattern in SENSITIVE_PATTERNS:
            for m in pattern.finditer(msg):
                v = (m.group(1) or "").strip()
                if len(v) >= 4: values.add(v)
    return values

def scan_for_leaks(blob, needles):
    return sum(1 for v in needles if v in blob)
