"""Start TechScope, preferring models that are already on disk.

Setup
-----
    python -m venv .venv
    .venv\\Scripts\\activate            # Linux/macOS: source .venv/bin/activate
    pip install -r requirements.txt
    python -m spacy download en_core_web_sm

Then either:

    streamlit run app.py                # fetches the sentence-transformer once
    python run_local.py                 # same, but see "Offline" below

Offline
-------
On a machine with no network, fetch both models elsewhere and drop them next to
this file:

    techscope/models/en_core_web_sm/
    techscope/models/all-MiniLM-L6-v2/

They are picked up automatically. Any other location works through the
environment instead:

    TECHSCOPE_SPACY_MODEL=/path/to/en_core_web_sm
    TECHSCOPE_EMBED_MODEL=/path/to/all-MiniLM-L6-v2

A larger spaCy model (en_core_web_md / _lg) parses more accurately and is worth
using if the disk space is there. Nothing leaves the machine either way.
"""
import os
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODELS = HERE / "models"

for var, folder in (("TECHSCOPE_SPACY_MODEL", "en_core_web_sm"),
                    ("TECHSCOPE_EMBED_MODEL", "all-MiniLM-L6-v2")):
    local = MODELS / folder
    if local.is_dir() and not os.environ.get(var):
        os.environ[var] = str(local)

# Torch and spaCy can each pull in their own OpenMP runtime; on Windows the
# duplicate aborts the process at import time rather than raising.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# On a machine that has never run Streamlit, `streamlit run` stops at an
# interactive "Email:" prompt and waits forever. Writing the empty credentials
# file is exactly what pressing Enter at that prompt does, and it keeps the
# launcher non-interactive without giving up the browser auto-open that
# --server.headless would cost.
creds = Path.home() / ".streamlit" / "credentials.toml"
if not creds.exists():
    try:
        creds.parent.mkdir(parents=True, exist_ok=True)
        creds.write_text('[general]\nemail = ""\n', encoding="utf-8")
    except OSError:
        pass                      # read-only home: fall through, prompt may appear

os.chdir(HERE)
sys.path.insert(0, str(HERE))
sys.argv = ["streamlit", "run", str(HERE / "app.py"),
            "--server.port", os.environ.get("TECHSCOPE_PORT", "8501"),
            "--browser.gatherUsageStats", "false"]
runpy.run_module("streamlit", run_name="__main__")
