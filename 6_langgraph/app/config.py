import os
from pathlib import Path

from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "gemini-3.5-flash-lite")
APPROVAL_TOTAL_THRESHOLD = os.environ.get("APPROVAL_TOTAL_THRESHOLD", "500")
GRAPH_RECURSION_LIMIT = os.environ.get("GRAPH_RECURSION_LIMIT", "25")
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*").split(",")

# --- Resilience knobs ---

# Max number of LLM call attempts (including the first) before the node fails.
LLM_RETRY_MAX_ATTEMPTS = int(os.environ.get("LLM_RETRY_MAX_ATTEMPTS", "3"))

# Seconds before the first retry; doubles each attempt (capped at 30 s).
LLM_RETRY_INITIAL_INTERVAL = float(os.environ.get("LLM_RETRY_INITIAL_INTERVAL", "1.0"))

# Hard wall-clock limit (seconds) for a single review_item LLM call.
# If the model hangs longer than this, NodeTimeoutError is raised and the
# retry policy decides whether to attempt again.
REVIEW_ITEM_TIMEOUT = float(os.environ.get("REVIEW_ITEM_TIMEOUT", "30.0"))

# Checkpoint durability mode passed to astream():
#   "sync"  – flush checkpoint before the next node starts (safest, slowest)
#   "async" – flush in background while the next node runs (default, balanced)
#   "exit"  – flush only when the graph exits (fastest, least safe)
DURABILITY_MODE = os.environ.get("DURABILITY_MODE", "async")