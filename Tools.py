"""
tools.py - Deterministic financial tools + guideline RAG helpers.
 
Design rule: NOTHING in this file calls an LLM. Every number the underwriter sees
(DTI, LTV, threshold status, document age) is computed here with exact Decimal math.
 
Plain functions are the source of truth (unit-testable). Thin CrewAI-wrapped
versions (`*_crew_tool`) are exposed at the bottom for agents.
 
IMPORTANT: PROGRAM_RULES and GUIDELINE_KB are illustrative, paraphrased defaults.
They are NOT a substitute for the current Fannie Mae Selling Guide, Freddie Mac
Guide, HUD 4000.1, VA Lenders Handbook, or your institution's overlays. Replace /
extend them with your approved policy corpus before any production use.
"""
from __future__ import annotations
 
import re
from datetime import date, datetime, timedelta
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any, Iterable, Optional
 
try:  # tools.py stays importable/testable without crewai installed
    from crewai.tools import tool as _crewai_tool
except Exception:  # pragma: no cover
    _crewai_tool = None
 
 
def _tool(name: str):
    def deco(fn):
        return _crewai_tool(name)(fn) if _crewai_tool else fn
    return deco
 
 
class InputValidationError(ValueError):
    """Raised for missing / non-numeric / negative / non-finite inputs."""
 
 
# --------------------------------------------------------------------------- #
# 1. DETERMINISTIC MATH
# --------------------------------------------------------------------------- #
def _to_decimal(label: str, value: Any, *, allow_zero: bool) -> Decimal:
    if value is None or isinstance(value, bool):
        raise InputValidationError(f"{label} is required.")
    try:
        d = Decimal(str(value).replace(",", "").replace("$", "").strip())
    except (InvalidOperation, ValueError):
        raise InputValidationError(f"{label} must be a number (got {value!r}).")
    if not d.is_finite():
        raise InputValidationError(f"{label} must be a finite number.")
    if d < 0:
        raise InputValidationError(f"{label} cannot be negative.")
    if d == 0 and not allow_zero:
        raise InputValidationError(f"{label} must be greater than zero.")
    return d
 
 
def _pct(numerator: Decimal, denominator: Decimal) -> float:
    return float((numerator / denominator * 100).quantize(Decimal("0.01"), ROUND_HALF_UP))
 
 
def calculate_dti_and_ltv(monthly_income, monthly_debts, loan_amount, property_value) -> dict:
    """
    Back-end DTI % = total monthly obligations (incl. proposed housing payment) / gross monthly income.
    LTV % = loan amount / property value (use the LOWER of appraised value and purchase price).
    Strict Decimal math, rounded half-up to 2 places. Raises InputValidationError on bad input.
    """
    income = _to_decimal("Monthly income", monthly_income, allow_zero=False)
    debts = _to_decimal("Monthly debts", monthly_debts, allow_zero=True)
    loan = _to_decimal("Loan amount", loan_amount, allow_zero=False)
    value = _to_decimal("Property value", property_value, allow_zero=False)
    return {
        "dti_percent": _pct(debts, income),
        "ltv_percent": _pct(loan, value),
        "inputs": {
            "monthly_income": float(income),
            "monthly_debts": float(debts),
            "loan_amount": float(loan),
            "property_value": float(value),
        },
    }
 
 
# --------------------------------------------------------------------------- #
# 1b. DETERMINISTIC RULE VALIDATION (configurable defaults - verify vs. current policy)
# --------------------------------------------------------------------------- #
PROGRAM_RULES: dict[str, dict] = {
    "Conventional 30-Yr Fixed": {
        "dti_review": 45.0, "dti_max": 50.0, "ltv_max": 97.0, "mi_ltv": 80.0, "always_mi": False,
        "basis": "Fannie Mae / Freddie Mac (DU/LPA ceiling 50% DTI; manual 36-45%; 97% max LTV 1-unit)",
    },
    "FHA": {
        "dti_review": 43.0, "dti_max": 56.9, "ltv_max": 96.5, "mi_ltv": None, "always_mi": True,
        "basis": "HUD 4000.1 (manual 31/43; AUS-approved up to ~56.9%; 96.5% max LTV at 580+)",
    },
    "VA": {
        "dti_review": 41.0, "dti_max": None, "ltv_max": 100.0, "mi_ltv": None, "always_mi": False,
        "basis": "VA guideline 41% DTI; above it, residual income + compensating factors govern",
    },
}
 
 
def evaluate_program_thresholds(program: str, dti_percent: float, ltv_percent: float) -> dict:
    """Deterministic rule validation. Returns status in {within, review, exceeds, unknown_program}."""
    rules = PROGRAM_RULES.get(program)
    if rules is None:
        return {"status": "unknown_program", "rules": None, "flags": [{
            "severity": "review", "code": "UNKNOWN_PROGRAM",
            "message": f"No rule set configured for program '{program}'. Manual review required."}]}
    flags: list[dict] = []
    if rules["dti_max"] is not None and dti_percent > rules["dti_max"]:
        flags.append({"severity": "exceeds", "code": "DTI_EXCEEDS_MAX",
                      "message": f"DTI {dti_percent:.2f}% exceeds program ceiling {rules['dti_max']:.1f}%."})
    elif dti_percent > rules["dti_review"]:
        extra = (" Residual income analysis + compensating factors required." if program == "VA"
                 else " Requires strong compensating factors / AUS approval.")
        flags.append({"severity": "review", "code": "DTI_ABOVE_REVIEW",
                      "message": f"DTI {dti_percent:.2f}% exceeds review level {rules['dti_review']:.1f}%." + extra})
    if ltv_percent > rules["ltv_max"]:
        flags.append({"severity": "exceeds", "code": "LTV_EXCEEDS_MAX",
                      "message": f"LTV {ltv_percent:.2f}% exceeds program maximum {rules['ltv_max']:.1f}%."})
    if rules["mi_ltv"] is not None and ltv_percent > rules["mi_ltv"]:
        flags.append({"severity": "info", "code": "MI_REQUIRED",
                      "message": f"LTV above {rules['mi_ltv']:.0f}% - mortgage insurance / enhancement required."})
    if rules["always_mi"]:
        flags.append({"severity": "info", "code": "MIP_APPLIES",
                      "message": "FHA mortgage insurance premium applies to this program."})
    sev = {f["severity"] for f in flags}
    status = "exceeds" if "exceeds" in sev else "review" if "review" in sev else "within"
    return {"status": status, "rules": rules, "flags": flags}
 
 
# --------------------------------------------------------------------------- #
# 3. DOCUMENT COMPLETENESS / FRESHNESS
# --------------------------------------------------------------------------- #
# Classification is first-match-wins, in this order (so "earnings statement" is a
# paystub, not a bank statement).
CHECKLIST: list[dict] = [
    {"category": "Income/W2", "label": "W-2 / tax return (most recent year)",
     "pattern": r"w-?2|1040|tax return|1099", "max_age_days": None},
    {"category": "Income/W2", "label": "Recent paystub (<=30 days)",
     "pattern": r"pay ?stub|pay stub|earnings statement|pay advice", "max_age_days": 30},
    {"category": "Credit/Debts", "label": "Credit report (<=120 days)",
     "pattern": r"credit", "max_age_days": 120},
    {"category": "Appraisal", "label": "Appraisal (<=120 days)",
     "pattern": r"apprais", "max_age_days": 120},
    {"category": "Assets/Bank Statements", "label": "Bank/asset statements (<=60 days)",
     "pattern": r"bank|checking|savings|asset|401\(?k\)?|brokerage|account statement", "max_age_days": 60},
]
 
 
def _parse_date(value: Any) -> Optional[date]:
    if value is None or value != value:  # None or NaN/NaT
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%d-%b-%Y", "%b %d, %Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None
 
 
def _normalize_docs(documents_provided: Iterable[Any]) -> list[dict]:
    docs = []
    for d in documents_provided or []:
        if isinstance(d, str):
            name, dt = d, None
        elif isinstance(d, dict):
            name = d.get("type") or d.get("name") or d.get("Document") or ""
            dt = d.get("date") if d.get("date") is not None else d.get("Date")
        else:
            continue
        name = str(name).strip()
        if name:
            docs.append({"name": name, "date": _parse_date(dt)})
    return docs
 
 
def check_document_completeness(documents_provided, reference_date: Optional[date] = None) -> dict:
    """
    Evaluate the file against the required checklist. Conservative by design:
    a time-sensitive document with no (or a future) date is 'undated' = cannot verify = NOT complete.
    documents_provided: list of str or dicts {"type"/"name": str, "date": date|str|None}
    """
    ref = reference_date or date.today()
    docs = _normalize_docs(documents_provided)
    buckets: dict[int, list[dict]] = {i: [] for i in range(len(CHECKLIST))}
    unclassified = []
    for doc in docs:
        for i, item in enumerate(CHECKLIST):
            if re.search(item["pattern"], doc["name"], re.I):
                buckets[i].append(doc)
                break
        else:
            unclassified.append(doc["name"])
 
    items = []
    for i, item in enumerate(CHECKLIST):
        found = buckets[i]
        row = {"category": item["category"], "label": item["label"], "newest_date": None,
               "age_days": None, "status": "missing", "detail": "No matching document in file."}
        if found:
            dated = [d for d in found if d["date"] is not None]
            newest = max(dated, key=lambda d: d["date"]) if dated else None
            if item["max_age_days"] is None:
                row.update(status="present", detail="Present (age not time-sensitive).",
                           newest_date=newest["date"].isoformat() if newest else None)
            elif newest is None:
                row.update(status="undated", detail="Document present but undated - age cannot be verified.")
            else:
                age = (ref - newest["date"]).days
                row.update(newest_date=newest["date"].isoformat(), age_days=age)
                if age < 0:
                    row.update(status="undated", detail="Document date is in the future - verify date.")
                elif age > item["max_age_days"]:
                    row.update(status="outdated",
                               detail=f"{age} days old; limit is {item['max_age_days']} days.")
                else:
                    row.update(status="present", detail=f"{age} days old (within {item['max_age_days']}).")
        items.append(row)
 
    problems = [r for r in items if r["status"] != "present"]
    return {
        "complete": not problems,
        "reference_date": ref.isoformat(),
        "items": items,
        "missing": [r["label"] for r in items if r["status"] == "missing"],
        "outdated": [r["label"] for r in items if r["status"] == "outdated"],
        "undated": [r["label"] for r in items if r["status"] == "undated"],
        "unclassified_documents": unclassified,
    }
 
 
# --------------------------------------------------------------------------- #
# 2. GUIDELINE RAG (local, dependency-free retrieval over a curated corpus)
# --------------------------------------------------------------------------- #
GUIDELINE_KB: list[dict] = [
    {"id": "FNMA-RESERVES", "agency": "Fannie Mae", "topic": "Reserve requirements",
     "section_hint": "Selling Guide B3-4.1 (asset verification) / DU message",
     "keywords": ["reserves", "reserve", "months", "liquid", "assets", "conventional", "fannie"],
     "text": "Required reserves are set by the DU findings message or, for manually underwritten loans, by a "
             "matrix driven by credit score, DTI, LTV, occupancy and units. Reserves must be verified, liquid "
             "and documented; retirement and non-liquid assets are subject to haircuts or limits."},
    {"id": "FNMA-GIFT", "agency": "Fannie Mae", "topic": "Gift funds",
     "section_hint": "Selling Guide B3-4.3-04 (personal gifts)",
     "keywords": ["gift", "gifts", "donor", "relative", "letter", "funds", "down", "payment", "conventional", "fannie"],
     "text": "Gifts must come from an eligible donor (e.g. relative, domestic partner, fiance) who has no "
             "interest in the transaction. A gift letter stating donor, relationship, amount and no repayment "
             "obligation is required, plus evidence of transfer of the funds."},
    {"id": "FNMA-LARGE-DEPOSIT", "agency": "Fannie Mae", "topic": "Large deposits / source of funds",
     "section_hint": "Selling Guide B3-4.2-02 (evaluating bank statements)",
     "keywords": ["deposit", "deposits", "large", "source", "sourcing", "bank", "statement", "unsourced", "funds"],
     "text": "On purchase transactions, a single deposit exceeding 50% of total monthly qualifying income is a "
             "large deposit and requires documentation of its source. Unsourced large deposits cannot be used "
             "for down payment, closing costs or reserves."},
    {"id": "FNMA-DTI", "agency": "Fannie Mae", "topic": "Debt-to-income limits",
     "section_hint": "Selling Guide B3-6-02 (DTI ratios)",
     "keywords": ["dti", "debt", "income", "ratio", "limit", "limits", "conventional", "fannie", "freddie", "compensating"],
     "text": "Manually underwritten loans are generally capped at 36% DTI, up to 45% with compensating factors "
             "and reserves. Loans underwritten through DU may be approved up to 50% DTI."},
    {"id": "FNMA-LTV-MI", "agency": "Fannie Mae", "topic": "LTV limits and mortgage insurance",
     "section_hint": "Selling Guide B2-1.2 (eligibility matrix)",
     "keywords": ["ltv", "loan", "value", "mortgage", "insurance", "mi", "conventional", "fannie", "freddie", "collateral"],
     "text": "Conventional loans above 80% LTV require mortgage insurance or equivalent credit enhancement. "
             "Maximum LTV depends on occupancy, units, purpose and loan type; the high end for a one-unit "
             "principal residence fixed-rate purchase is 97%."},
    {"id": "FNMA-INCOME-STABILITY", "agency": "Fannie Mae", "topic": "Income stability and continuity",
     "section_hint": "Selling Guide B3-3.1 (income assessment)",
     "keywords": ["income", "stable", "stability", "continuity", "continue", "history", "employment", "paystub", "w2", "w-2"],
     "text": "Income must be stable, predictable and likely to continue. Employment income is generally "
             "supported by a two-year history; non-employment income (e.g. alimony, Social Security) must "
             "generally continue at least three years. Gaps and declining trends must be analyzed."},
    {"id": "FNMA-VOE-AGE", "agency": "Fannie Mae", "topic": "Employment verification timing & document age",
     "section_hint": "Selling Guide B3-3.1-07 / B1-1-03 (allowable age of documents)",
     "keywords": ["voe", "verification", "employment", "verbal", "age", "document", "documents", "outdated", "stale",
                  "credit", "appraisal", "paystub", "days", "months", "old"],
     "text": "Verbal verification of employment is required shortly before closing (within 10 business days of "
             "the note date for salaried borrowers). Credit and appraisal documents have a maximum allowable "
             "age (generally four months at note date). Paystubs must be recent (generally within 30 days of "
             "application) and asset statements must cover the most recent period (typically two months)."},
    {"id": "FNMA-SELF-EMPLOYED", "agency": "Fannie Mae", "topic": "Self-employed income",
     "section_hint": "Selling Guide B3-3.2 (self-employed)",
     "keywords": ["self", "employed", "self-employed", "business", "tax", "returns", "schedule", "k-1", "1040", "profit"],
     "text": "Self-employed income is generally documented with two years of personal and (when ownership is "
             "25% or more) business tax returns, with analysis of trend, liquidity and business viability."},
    {"id": "GEN-DISCREPANCY", "agency": "Fannie Mae / Freddie Mac (general)", "topic": "Resolving file inconsistencies",
     "section_hint": "Underwriter due-diligence principle",
     "keywords": ["discrepancy", "mismatch", "inconsistent", "inconsistency", "employer", "name", "undisclosed", "debt",
                  "debts", "liability", "liabilities", "explanation", "letter", "loe", "credit", "report", "1003"],
     "text": "Material inconsistencies (employer name, income amount, undisclosed liabilities, occupancy) must "
             "be resolved with documentation or a written explanation before the file can be approved. "
             "Undisclosed obligations on the credit report must be included in DTI unless properly excluded."},
    {"id": "FRED-ASSETS", "agency": "Freddie Mac", "topic": "Asset documentation",
     "section_hint": "Freddie Mac Guide, assets chapter - verify current section",
     "keywords": ["freddie", "assets", "asset", "bank", "statements", "consecutive", "months", "reserves", "conventional"],
     "text": "Assets used to qualify must be documented with recent consecutive statements (generally two "
             "months) or a verification of deposit; unusual or large deposits must be explained and sourced."},
    {"id": "FHA-DTI-CREDIT", "agency": "FHA / HUD", "topic": "FHA qualifying ratios and credit score",
     "section_hint": "HUD 4000.1 underwriting (TOTAL scorecard / manual)",
     "keywords": ["fha", "dti", "ratio", "credit", "score", "580", "500", "manual", "total", "scorecard", "ltv", "minimum"],
     "text": "FHA manual underwriting baseline ratios are 31% housing / 43% total, with higher ratios only with "
             "documented compensating factors; AUS (TOTAL) approvals can go higher. A 580+ score supports "
             "maximum financing (96.5% LTV); 500-579 is limited to 90% LTV."},
    {"id": "FHA-GIFT", "agency": "FHA / HUD", "topic": "FHA gift funds",
     "section_hint": "HUD 4000.1 borrower assets - gifts",
     "keywords": ["fha", "gift", "gifts", "donor", "letter", "family", "employer", "charitable", "funds"],
     "text": "Gift funds may come from family members, employers, labor unions, charitable organizations or "
             "government agencies. A gift letter and documented transfer of funds are required."},
    {"id": "FHA-LARGE-DEPOSIT", "agency": "FHA / HUD", "topic": "FHA large deposits",
     "section_hint": "HUD 4000.1 borrower assets - large deposits",
     "keywords": ["fha", "deposit", "deposits", "large", "source", "bank", "statement", "explanation", "funds"],
     "text": "For manually underwritten loans, large increases in account balances (commonly cited as more than "
             "1% of the adjusted value) require a documented explanation and source. AUS loans follow the AUS findings."},
    {"id": "VA-RESIDUAL", "agency": "VA", "topic": "VA residual income and DTI",
     "section_hint": "VA Lenders Handbook (Chapter 4)",
     "keywords": ["va", "residual", "income", "dti", "41", "ratio", "compensating", "veteran", "entitlement"],
     "text": "VA uses a 41% DTI guideline together with residual income tests. Ratios above 41% require residual "
             "income at least 20% above the guideline or other compensating factors, documented by the underwriter."},
    {"id": "REGB-ADVERSE", "agency": "CFPB / ECOA (Reg B)", "topic": "Adverse action notices",
     "section_hint": "12 CFR 1002.9; CFPB Circular 2022-03",
     "keywords": ["adverse", "action", "denial", "deny", "reasons", "notice", "ecoa", "regulation", "cfpb", "specific",
                  "override", "decline"],
     "text": "A creditor must notify the applicant of adverse action within 30 days of a completed application and "
             "state the specific principal reasons. Generic statements are insufficient, and use of complex "
             "models or AI does not excuse the duty to give accurate, specific reasons."},
]
 
_STOP = {"the", "a", "an", "of", "to", "for", "and", "or", "in", "on", "is", "are", "what", "how", "do", "does",
         "with", "by", "at", "be", "it", "this", "that", "from", "as", "s"}
MIN_RELEVANCE_SCORE = 3
 
 
def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", (text or "").lower())
 
 
def guideline_search_tool(query: str, top_k: int = 4) -> list[dict]:
    """
    Retrieve the most relevant guideline entries. Keyword hits are weighted 3x over body hits.
    Returns [] when nothing clears MIN_RELEVANCE_SCORE - callers MUST treat [] as
    'Insufficient Information' rather than guessing a policy.
    """
    q = {t for t in _tokens(query) if t not in _STOP}
    q |= {p for t in list(q) for p in t.split("-") if p and p not in _STOP}
    if not q:
        return []
    scored = []
    for entry in GUIDELINE_KB:
        kw = set(entry["keywords"])
        body = set(_tokens(entry["topic"] + " " + entry["text"]))
        score = 3 * len(q & kw) + len(q & body)
        if score >= MIN_RELEVANCE_SCORE:
            scored.append({**entry, "score": score})
    scored.sort(key=lambda e: e["score"], reverse=True)
    return scored[:top_k]
 
 
# --------------------------------------------------------------------------- #
# Text formatters (feed deterministic output to agents / UI)
# --------------------------------------------------------------------------- #
def format_guideline_hits(hits: list[dict]) -> str:
    if not hits:
        return "INSUFFICIENT INFORMATION: no matching guideline found in the policy corpus for this query."
    lines = []
    for h in hits:
        lines.append(f"[{h['id']}] {h['agency']} - {h['topic']} ({h['section_hint']}): {h['text']}")
    lines.append("NOTE: Paraphrased reference text. Verify against the current official guide before relying on it.")
    return "\n".join(lines)
 
 
def format_completeness(c: dict) -> str:
    lines = [f"Document completeness (as of {c['reference_date']}): "
             f"{'COMPLETE' if c['complete'] else 'INCOMPLETE'}"]
    for r in c["items"]:
        lines.append(f"- {r['label']}: {r['status'].upper()} - {r['detail']}")
    if c["unclassified_documents"]:
        lines.append("- Unclassified documents: " + ", ".join(c["unclassified_documents"]))
    return "\n".join(lines)
 
 
def format_metrics(m: dict, t: dict, program: str) -> str:
    lines = [f"Program: {program}",
             f"DTI (deterministic): {m['dti_percent']:.2f}%",
             f"LTV (deterministic): {m['ltv_percent']:.2f}%",
             f"Rule validation status: {t['status'].upper()}"]
    for f in t["flags"]:
        lines.append(f"- [{f['severity'].upper()}] {f['message']}")
    return "\n".join(lines)
 
 
# --------------------------------------------------------------------------- #
# CrewAI tool wrappers (string in / string out)
# --------------------------------------------------------------------------- #
@_tool("Guideline Search Tool")
def guideline_search_crew_tool(query: str) -> str:
    """Search Fannie Mae / Freddie Mac / FHA / VA / Reg B policy excerpts. Input: a short topic query such as
    'gift funds documentation' or 'FHA large deposit'. Returns policy excerpts with IDs, or
    'INSUFFICIENT INFORMATION' if nothing relevant exists. Never guess policy that is not returned."""
    return format_guideline_hits(guideline_search_tool(query))
 
 
@_tool("DTI and LTV Calculator")
def calculate_dti_and_ltv_crew_tool(monthly_income: float, monthly_debts: float,
                                    loan_amount: float, property_value: float) -> str:
    """Exact DTI% and LTV% calculator. Inputs: gross monthly income, total monthly debts (including proposed
    housing payment), loan amount, property value. Always use this instead of mental math."""
    try:
        r = calculate_dti_and_ltv(monthly_income, monthly_debts, loan_amount, property_value)
        return f"DTI = {r['dti_percent']:.2f}% ; LTV = {r['ltv_percent']:.2f}%"
    except InputValidationError as e:
        return f"INSUFFICIENT INFORMATION: {e}"
 
 
@_tool("Document Completeness Checker")
def check_document_completeness_crew_tool(documents: str) -> str:
    """Check borrower file completeness. Input: one document per line as 'name | YYYY-MM-DD'
    (date optional). Returns missing / outdated / undated required documents."""
    docs = []
    for line in (documents or "").splitlines():
        if not line.strip():
            continue
        name, _, dt = line.partition("|")
        docs.append({"name": name.strip(), "date": dt.strip() or None})
    return format_completeness(check_document_completeness(docs))
 
