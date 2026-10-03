"""
app.py - Mortgage Underwriting Co-Pilot (Streamlit).
Run:  streamlit run app.py
"""
from __future__ import annotations
 
import os
 
# Privacy: mortgage data must not leave the app via framework telemetry.
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
 
import sys
 
# Streamlit Cloud ships an old sqlite3; crewai's vector-store dependency (chromadb) needs >= 3.35.
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    pass
 
import importlib.metadata as md
import sqlite3
import traceback
from datetime import date, datetime, timedelta
 
import pandas as pd
import streamlit as st
 
st.set_page_config(page_title="Underwriting Co-Pilot", page_icon="🏦", layout="wide")  # must be first st call
 
try:  # optional: Streamlit Cloud uses Secrets instead of a .env file
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    def load_dotenv(*args, **kwargs):
        return False
 
APP_DIR = os.path.dirname(os.path.abspath(__file__))
try:
    from agents import REC_ALL, REC_APPROVE, REC_DENY, REC_SUSPEND, WorkflowError, run_underwriting_workflow
    from tools import PROGRAM_RULES, InputValidationError
except Exception as _exc:  # Streamlit redacts tracebacks, so surface the real cause ourselves
    st.error("The app failed to start because a module could not be imported.")
    st.code(traceback.format_exc())
    st.write("Files next to app.py:", sorted(os.listdir(APP_DIR)))
    st.info("Make sure agents.py, tools.py and requirements.txt are in the same folder as app.py in your GitHub repo, "
            "then reboot the app. See README.md.")
    st.stop()
 
load_dotenv()
 
 
def get_secret(name: str) -> str:
    """Streamlit Secrets first (Cloud), then environment / .env (local)."""
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return os.getenv(name, "")
 
 
ADVERSE_REASONS = [
    "Excessive obligations in relation to income",
    "Unable to verify income",
    "Unable to verify employment",
    "Unable to verify credit references",
    "Unable to verify source of funds",
    "Insufficient funds for down payment / closing / reserves",
    "Value or type of collateral not sufficient",
    "Delinquent past or present credit obligations",
    "Other (specify in details)",
]
MIN_OVERRIDE_CHARS = 20
DOC_COLUMNS = ["Document", "Date"]
 
 
# ----------------------------- state helpers ------------------------------ #
def empty_docs() -> pd.DataFrame:
    return pd.DataFrame({"Document": pd.Series(dtype="str"), "Date": pd.Series(dtype="datetime64[ns]")})
 
 
def init_state():
    defaults = {"income": 0.0, "debts": 0.0, "loan": 0.0, "value": 0.0, "notes": "",
                "docs_df": empty_docs(), "docs_version": 0, "docs_current": empty_docs(),
                "result": None, "final": None, "audit": [], "pending": None}
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)
 
 
def log_event(event: str, detail: str = ""):
    st.session_state.audit.append({"timestamp": datetime.now().isoformat(timespec="seconds"),
                                   "event": event, "detail": detail})
 
 
def load_sample():
    t = date.today()
    st.session_state.update(
        income=9500.0, debts=3650.0, loan=340000.0, value=400000.0,
        notes=("1003 lists gross monthly income of $9,500. Most recent paystub YTD annualizes to about $8,900/month.\n"
               "Employer on paystub: Northwind Logistics LLC. Employer on 2025 W-2: North Wind Logistic Inc.\n"
               "Credit report shows an auto loan ($412/mo) and a student loan ($186/mo); the 1003 lists only the auto loan.\n"
               "Most recent bank statement shows a $10,000 deposit with no source documentation.\n"
               "Borrower states $5,000 of the down payment is a gift from an uncle; no gift letter is in the file."),
        docs_df=pd.DataFrame({
            "Document": ["Paystub", "W-2 (2025)", "Credit report", "Bank statement (checking)", "Appraisal"],
            "Date": pd.to_datetime([t - timedelta(days=12), t - timedelta(days=200), t - timedelta(days=30),
                                    t - timedelta(days=20), t - timedelta(days=150)])}),
        result=None, final=None, pending=None)
    st.session_state.docs_version += 1
    st.session_state.audit = []
    log_event("sample_loaded", "Demo file with planted discrepancies loaded.")
 
 
def docs_to_list(df: pd.DataFrame) -> list[dict]:
    out = []
    for _, row in df.iterrows():
        name = str(row.get("Document") or "").strip()
        if not name or name.lower() == "nan":
            continue
        d = row.get("Date")
        out.append({"name": name, "date": None if pd.isna(d) else pd.Timestamp(d).date()})
    return out
 
 
# ------------------------------ report builder ---------------------------- #
def build_report(res: dict, final: dict, audit: list[dict]) -> str:
    m, d, c, t = res["metrics"], res["decision"], res["completeness"], res["thresholds"]
    L = ["# Underwriting Decision Summary", "",
         f"- **Generated:** {datetime.now().isoformat(timespec='seconds')}",
         f"- **Program:** {res['program']}",
         f"- **Analysis run:** {res['run_timestamp']} (model: {res['model']})",
         f"- **Underwriter of record:** {final['underwriter']}", "",
         "## Final Human Decision", f"**{final['final_decision']}**  (action: {final['action']})", ""]
    if final.get("override_reason"):
        L += [f"**Override reason (documented):** {final['override_reason']}", ""]
    if final.get("comment"):
        L += [f"**Underwriter comment:** {final['comment']}", ""]
    if final.get("conditions"):
        L += ["### Conditions / Information Requested", final["conditions"], ""]
    if final.get("adverse_reasons") or final.get("adverse_details"):
        L += ["### Adverse Action - Principal Reasons (log for ECOA / Reg B notice)"]
        L += [f"- {r}" for r in final.get("adverse_reasons", [])]
        if final.get("adverse_details"):
            L += ["", f"Details: {final['adverse_details']}"]
        L += ["", "_Notice must be sent within 30 days of a completed application (12 CFR 1002.9)._", ""]
    L += ["## Deterministic Metrics",
          f"- DTI: **{m['dti_percent']:.2f}%**  |  LTV: **{m['ltv_percent']:.2f}%**",
          f"- Rule validation: **{t['status'].upper()}**"]
    L += [f"  - [{f['severity']}] {f['message']}" for f in t["flags"]]
    L += ["", "## AI Decision Support Matrix (advisory only)",
          f"- **AI recommendation (after guardrails):** {d['recommendation']}",
          f"- **AI original recommendation:** {d['ai_original_recommendation']}",
          f"- **Rationale:** {d['recommendation_rationale']}", "", "### Supporting Findings"]
    L += [f"- {x}" for x in d["supporting_findings"]] or ["- None identified"]
    L += ["", "### Unresolved Issues / Discrepancies"]
    L += [f"- {x}" for x in d["unresolved_issues"]] or ["- None identified"]
    L += ["", "### Suggested Conditions"]
    L += [f"- {x['condition']}  \n  _Evidence:_ {x['evidence_basis']}  \n  _Guideline:_ {x['guideline_reference']}"
          for x in d["suggested_conditions"]] or ["- None"]
    if d["insufficient_information"]:
        L += ["", "### INSUFFICIENT INFORMATION"] + [f"- {r}" for r in d["insufficient_information_reasons"]]
    L += ["", "## Document Completeness"] + [f"- {r['label']}: {r['status'].upper()} - {r['detail']}" for r in c["items"]]
    L += ["", "## Guideline Citations Retrieved"] + [
        f"- [{h['id']}] {h['agency']} - {h['topic']} ({h['section_hint']})" for h in res["citations"]]
    if res["guardrail_log"]:
        L += ["", "## Guardrail Log"] + [f"- {x}" for x in res["guardrail_log"]]
    L += ["", "## Audit Trail"] + [f"- {a['timestamp']} - {a['event']}: {a['detail']}" for a in audit]
    L += ["", "---", "_AI output is decision support only; the underwriter holds final authority. Guideline text in this "
          "tool is paraphrased - verify against current official guides before relying on it._"]
    return "\n".join(L)
 
 
# --------------------------------- sidebar -------------------------------- #
init_state()
with st.sidebar:
    st.header("⚙️ Configuration")
    env_key = get_secret("GROQ_API_KEY")
    api_key = st.text_input("Groq API key", value=env_key, type="password",
                            help="Loaded from .env if present. Never stored by this app.")
    if env_key and api_key == env_key:
        st.caption("Key loaded from environment.")
    program = st.selectbox("Loan program", list(PROGRAM_RULES.keys()))
    r = PROGRAM_RULES[program]
    st.caption(f"Rules: {r['basis']}")
    with st.expander("Environment diagnostics"):
        for pkg in ("streamlit", "crewai", "litellm", "pydantic", "pandas", "python-dotenv"):
            try:
                st.write(f"✅ {pkg} {md.version(pkg)}")
            except Exception:
                st.write(f"❌ {pkg} not installed")
        st.caption(f"Python {sys.version.split()[0]} · sqlite {sqlite3.sqlite_version}")
    st.divider()
    st.button("Load sample file (demo)", on_click=load_sample)
    st.warning("Do not enter real borrower PII unless your LLM vendor agreement permits it.", icon="🔒")
 
st.title("🏦 Mortgage Underwriting Co-Pilot")
st.caption("AI reads and explains. Code calculates and validates. The underwriter decides.")
tab1, tab2, tab3 = st.tabs(["1 · Borrower Package", "2 · Automated Analysis", "3 · Underwriter Decision"])
 
# ---------------------------------- TAB 1 -------------------------------- #
with tab1:
    c1, c2 = st.columns(2)
    with c1:
        st.number_input("Gross monthly income ($)", min_value=0.0, step=100.0, format="%.2f", key="income")
        st.number_input("Total monthly debts incl. proposed housing payment ($)", min_value=0.0, step=50.0,
                        format="%.2f", key="debts")
    with c2:
        st.number_input("Loan amount ($)", min_value=0.0, step=1000.0, format="%.2f", key="loan")
        st.number_input("Property value ($, lower of appraisal / price)", min_value=0.0, step=1000.0,
                        format="%.2f", key="value")
    st.subheader("Document inventory")
    st.caption("List each document in the file with its date. Missing, outdated or undated items are flagged deterministically.")
    st.session_state.docs_current = st.data_editor(
        st.session_state.docs_df, num_rows="dynamic",
        key=f"docs_editor_{st.session_state.docs_version}",
        column_config={"Document": st.column_config.TextColumn("Document", required=True),
                       "Date": st.column_config.DateColumn("Document date", format="YYYY-MM-DD")})
    st.subheader("Borrower package file notes / document excerpts")
    st.text_area("Paste excerpts from paystubs, W-2s, credit report, bank statements, 1003, LOEs...",
                 height=220, key="notes")
 
# ---------------------------------- TAB 2 -------------------------------- #
with tab2:
    if st.button("▶ Run Agentic Workflow", type="primary"):
        problems = []
        if st.session_state.income <= 0: problems.append("Monthly income must be greater than zero.")
        if st.session_state.loan <= 0: problems.append("Loan amount must be greater than zero.")
        if st.session_state.value <= 0: problems.append("Property value must be greater than zero.")
        if st.session_state.notes.strip() and not api_key.strip():
            problems.append("Groq API key is missing (sidebar, Streamlit Secrets or .env).")
        if problems:
            for p in problems:
                st.error(p)
        else:
            try:
                with st.spinner("Computing metrics, validating rules, and running the agent crew..."):
                    res = run_underwriting_workflow(
                        api_key=api_key, program=program,
                        monthly_income=st.session_state.income, monthly_debts=st.session_state.debts,
                        loan_amount=st.session_state.loan, property_value=st.session_state.value,
                        notes=st.session_state.notes, documents=docs_to_list(st.session_state.docs_current))
                st.session_state.result, st.session_state.final, st.session_state.pending = res, None, None
                st.session_state.audit = [a for a in st.session_state.audit if a["event"] == "sample_loaded"]
                log_event("analysis_run", f"program={program}; model={res['model']}")
                for g in res["guardrail_log"]:
                    log_event("guardrail", g)
                st.success("Analysis complete. Review it below, then go to the Decision tab.")
            except InputValidationError as e:
                st.error(f"Invalid input: {e}")
            except WorkflowError as e:
                st.error(str(e))
            except Exception as e:  # last-resort: never show a stack trace to the underwriter
                st.error(f"Unexpected error: {type(e).__name__}: {e}")
 
    res = st.session_state.result
    if not res:
        st.info("Enter the package in Tab 1 and run the workflow.")
    else:
        m, t = res["metrics"], res["thresholds"]
        rules = t.get("rules") or {}
        k1, k2, k3 = st.columns(3)
        dti_ref = rules.get("dti_review")
        k1.metric("DTI (deterministic)", f"{m['dti_percent']:.2f}%",
                  delta=f"{m['dti_percent'] - dti_ref:+.2f} pts vs review level" if dti_ref else None,
                  delta_color="inverse")
        k2.metric("LTV (deterministic)", f"{m['ltv_percent']:.2f}%",
                  delta=f"{m['ltv_percent'] - rules['ltv_max']:+.2f} pts vs max" if rules else None,
                  delta_color="inverse")
        k3.metric("Rule validation", t["status"].upper())
        for f in t["flags"]:
            {"exceeds": st.error, "review": st.warning, "info": st.info}[f["severity"]](f["message"])
 
        st.subheader("Document completeness")
        st.dataframe(pd.DataFrame(res["completeness"]["items"])[["category", "label", "status", "newest_date", "detail"]],
                     hide_index=True)
 
        st.subheader("Guideline citations (RAG)")
        if not res["citations"]:
            st.warning("INSUFFICIENT INFORMATION: no matching guidelines retrieved.")
        for h in res["citations"]:
            with st.expander(f"[{h['id']}] {h['agency']} - {h['topic']}"):
                st.write(h["text"])
                st.caption(f"Reference: {h['section_hint']} · paraphrased - verify against the official guide.")
        with st.expander("Agent 1 - Document & Discrepancy Analysis"):
            st.markdown(res["analysis_text"])
        with st.expander("Agent 2 - Policy & Compliance Mapping"):
            st.markdown(res["policy_text"])
 
# ---------------------------------- TAB 3 -------------------------------- #
with tab3:
    res = st.session_state.result
    if not res:
        st.info("Run the analysis first.")
        st.stop()
    d = res["decision"]
 
    if d["insufficient_information"]:
        st.error("⚠️ **INSUFFICIENT INFORMATION - CFPB safeguard active.** The system will not approve or deny on "
                 "missing or inconclusive evidence.\n\n" + "\n".join(f"- {r}" for r in d["insufficient_information_reasons"]))
    if res["guardrail_log"]:
        with st.expander("🛡️ Guardrail log (deterministic overrides of AI output)", expanded=d["recommendation"] != d["ai_original_recommendation"]):
            for g in res["guardrail_log"]:
                st.write("• " + g)
 
    st.subheader("Review Matrix")
    st.markdown(f"### Recommendation: `{d['recommendation']}`")
    st.caption(d["recommendation_rationale"])
    a, b, c = st.columns(3)
    with a:
        st.markdown("**Supporting findings**")
        for x in d["supporting_findings"] or ["None identified"]: st.write("• " + x)
    with b:
        st.markdown("**Unresolved issues / discrepancies**")
        for x in d["unresolved_issues"] or ["None identified"]: st.write("• " + x)
    with c:
        st.markdown("**Suggested conditions**")
        for x in d["suggested_conditions"] or []:
            st.write(f"• {x['condition']}")
            st.caption(f"Evidence: {x['evidence_basis']} · Guideline: {x['guideline_reference']}")
        if not d["suggested_conditions"]: st.write("• None")
 
    st.divider()
    st.subheader("Action Station")
    if st.session_state.final:
        st.success(f"Decision recorded: **{st.session_state.final['final_decision']}**")
    else:
        b1, b2, b3 = st.columns(3)
        if b1.button("✅ Accept Recommendation"): st.session_state.pending = "accept"
        if b2.button("📝 Request Information / Add Conditions"): st.session_state.pending = "request"
        if b3.button("⛔ Override Decision"): st.session_state.pending = "override"
 
        pending = st.session_state.pending
        default_aa = [x for x in res["adverse_action_candidates"] if x in ADVERSE_REASONS]
 
        if pending:
            with st.form(f"form_{pending}"):
                uw = st.text_input("Underwriter name / ID (required for audit)")
                comment = override_reason = conditions = ""
                final_decision = d["recommendation"]
                if pending == "accept":
                    st.write(f"Accepting: **{d['recommendation']}**")
                    comment = st.text_area("Optional comment")
                elif pending == "request":
                    final_decision = "Conditions Requested - Pending Borrower Response"
                    conditions = st.text_area(
                        "Conditions / information requests (editable)", height=160,
                        value="\n".join(f"{i}. {x['condition']}" for i, x in enumerate(d["suggested_conditions"], 1)))
                else:
                    options = [x for x in REC_ALL if x != d["recommendation"]]
                    final_decision = st.selectbox("Your final decision", options)
                    override_reason = st.text_area(
                        f"Override reason (REQUIRED, min {MIN_OVERRIDE_CHARS} chars - feeds adverse action logging)", height=120)
                aa_reasons = st.multiselect("Adverse action principal reasons (required if final decision is a denial)",
                                            ADVERSE_REASONS, default=default_aa)
                aa_details = st.text_area("Adverse action details (specific facts behind the reasons)", height=80)
                submitted = st.form_submit_button("Confirm & record decision", type="primary")
 
            if submitted:
                errs = []
                if not uw.strip(): errs.append("Underwriter name / ID is required.")
                if pending == "override" and len(override_reason.strip()) < MIN_OVERRIDE_CHARS:
                    errs.append(f"An override reason of at least {MIN_OVERRIDE_CHARS} characters is required.")
                if pending == "request" and not conditions.strip():
                    errs.append("Enter at least one condition or information request.")
                is_denial = final_decision == REC_DENY
                if is_denial and not aa_reasons:
                    errs.append("A denial requires at least one specific adverse action reason.")
                if is_denial and "Other (specify in details)" in aa_reasons and not aa_details.strip():
                    errs.append("Specify details for the 'Other' adverse action reason.")
                if errs:
                    for e in errs: st.error(e)
                else:
                    st.session_state.final = {
                        "action": {"accept": "accepted AI recommendation", "request": "requested information / conditions",
                                   "override": "OVERRODE AI recommendation"}[pending],
                        "final_decision": final_decision, "underwriter": uw.strip(),
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "comment": comment.strip(), "override_reason": override_reason.strip(),
                        "conditions": conditions.strip(),
                        "adverse_reasons": aa_reasons if is_denial else [],
                        "adverse_details": aa_details.strip() if is_denial else ""}
                    log_event(f"decision_{pending}", f"{uw.strip()} -> {final_decision}"
                              + (f" | reason: {override_reason.strip()}" if override_reason.strip() else ""))
                    st.session_state.pending = None
                    st.rerun()
 
    if st.session_state.final:
        st.divider()
        st.subheader("Final Decision Summary Report")
        report = build_report(res, st.session_state.final, st.session_state.audit)
        st.download_button("⬇️ Download report (Markdown)", report, file_name=f"underwriting_decision_{date.today()}.md",
                           mime="text/markdown", type="primary")
        with st.expander("Preview"):
            st.markdown(report)
 
