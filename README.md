Mortgage Underwriting Co-Pilot

Hybrid-responsibility underwriting assistant: AI reads and explains, deterministic code calculates and validates, the underwriter decides. When evidence is missing the system returns Insufficient Information instead of guessing.

Repo layout (everything at the repo root)
├── app.py                      # Streamlit UI (entry point)
├── agents.py                   # CrewAI agents + deterministic guardrails
├── tools.py                    # DTI/LTV math, rule validation, document checks, guideline RAG
├── requirements.txt
├── .env.example                # local dev only
├── .gitignore
├── .streamlit/
│   ├── config.toml
│   └── secrets.toml.example    # template for Streamlit Cloud secrets
└── tests/test_core.py
Run locally
bash
python -m venv .venv && source .venv/bin/activate      # Python 3.11 or 3.12
pip install -r requirements.txt
cp .env.example .env                                    # add GROQ_API_KEY
streamlit run app.py
python -m pytest -q                                     # optional: 9 tests, no API key needed
Deploy to Streamlit Community Cloud via GitHub
Push all files above to the repo root (app.py and requirements.txt side by side). Never commit .env.
share.streamlit.io -> Create app -> pick the repo, branch, main file app.py.
Advanced settings -> Python version: 3.12 (crewai needs 3.10-3.13; 3.12 is the safest).
Advanced settings -> Secrets, paste:
toml
   GROQ_API_KEY = "your_groq_key"
   OPENAI_API_KEY = "your_openai_key"
Deploy. After any dependency change use Manage app -> Reboot.
Troubleshooting
Symptom	Cause	Fix
ModuleNotFoundError on dotenv, agents, crewai	requirements.txt not at repo root, or agents.py/tools.py not committed	Move files to repo root, commit, reboot. The app now prints the real traceback and a file listing on screen.
Sidebar -> Environment diagnostics shows ❌ crewai	crewai failed to install	Check Manage app logs; set Python 3.12; reboot. The math/rules/document checks still work without it.
sqlite3 >= 3.35 error	Old system sqlite on Cloud	Already handled via pysqlite3-binary (requirements) + swap in app.py. Needs Python < 3.13.
"Missing API key"	Secrets not set	Settings -> Secrets (see step 4) or type the key in the sidebar.
Safety design
DTI, LTV, threshold status and document age are computed in code (exact Decimal math), never by the LLM.
apply_guardrails() runs after the agents: it blocks approvals when evidence gaps exist or ratios exceed program maximums, and blocks denials that lack specific, evidence-backed reasons (Reg B).
Overrides require a documented reason; denials require specific principal reasons; every action is audit-logged.
Framework telemetry is disabled. Do not enter real borrower PII unless your LLM vendor agreement permits it.
PROGRAM_RULES and GUIDELINE_KB in tools.py are paraphrased illustrative defaults: replace them with your institution's approved policy corpus before production use.
