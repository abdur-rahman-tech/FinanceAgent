"""Run: python -m pytest -q   (needs only pydantic + pytest; crewai is NOT required)."""
import os, sys
from datetime import date, timedelta
 
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import agents, tools  # noqa: E402
 
T = date.today()
FRESH = [{"name": n, "date": T - timedelta(days=5)} for n in ("Paystub", "W-2", "Credit report", "Appraisal", "Bank statement")]
 
 
def test_exact_math():
    r = tools.calculate_dti_and_ltv(9500, 3650, 340000, 400000)
    assert (r["dti_percent"], r["ltv_percent"]) == (38.42, 85.0)
 
 
def test_bad_inputs_rejected():
    for bad in (0, -1, "abc", None, float("nan")):
        try:
            tools.calculate_dti_and_ltv(bad, 1, 1, 1)
            assert False, bad
        except tools.InputValidationError:
            pass
 
 
def test_stale_and_undated_docs_flagged():
    docs = [{"name": "Appraisal", "date": T - timedelta(days=200)}, {"name": "Paystub", "date": None}]
    c = tools.check_document_completeness(docs)
    assert not c["complete"] and "Appraisal (<=120 days)" in c["outdated"] and "Recent paystub (<=30 days)" in c["undated"]
 
 
def test_retrieval_returns_nothing_when_irrelevant():
    assert tools.guideline_search_tool("zebra quantum") == []
 
 
def test_guardrail_blocks_approval_on_gaps():
    c = tools.check_document_completeness([])
    th = tools.evaluate_program_thresholds("FHA", 30, 90)
    d, _ = agents.apply_guardrails({"recommendation": "Consider Conditional Approval"}, completeness=c, thresholds=th, notes_provided=True)
    assert d["recommendation"] == agents.REC_SUSPEND and d["insufficient_information"]
 
 
def test_guardrail_blocks_denial_without_reasons():
    c = tools.check_document_completeness(FRESH)
    th = tools.evaluate_program_thresholds("FHA", 30, 90)
    d, _ = agents.apply_guardrails({"recommendation": "Denial"}, completeness=c, thresholds=th, notes_provided=True)
    assert d["recommendation"] == agents.REC_SUSPEND
 
 
def test_dti_over_ceiling_never_approved():
    c = tools.check_document_completeness(FRESH)
    th = tools.evaluate_program_thresholds("Conventional 30-Yr Fixed", 55, 85)
    d, _ = agents.apply_guardrails({"recommendation": "Approve"}, completeness=c, thresholds=th, notes_provided=True)
    assert d["recommendation"] == agents.REC_DENY
 
 
def test_workflow_without_notes_never_calls_llm():
    r = agents.run_underwriting_workflow(api_key="", program="VA", monthly_income=8000, monthly_debts=2000,
                                         loan_amount=300000, property_value=320000, notes="", documents=[])
    assert r["model"] == "not invoked" and r["decision"]["recommendation"] == agents.REC_SUSPEND
 
 
def test_missing_key_gives_clean_error():
    try:
        agents.run_underwriting_workflow(api_key="", program="VA", monthly_income=8000, monthly_debts=2000,
                                         loan_amount=300000, property_value=320000, notes="x", documents=[])
        assert False
    except agents.WorkflowError as e:
        assert "API key" in str(e)
 
