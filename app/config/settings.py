"""
app/config/settings.py
-----------------------
All environment-driven configuration in one place. Component code reads these
constants; it never reads os.environ directly. Values may be supplied through
environment variables or a local .env file in the project root.

Why the API key is read with a default of "" rather than crashing on import:
    Importing this module (and therefore the gateway module) must be safe even
    when no key is set — tests import the gateway package to use the FakeGateway,
    which needs no key. The real OpenAIGateway validates the key in its
    constructor and fails clearly there. So a missing key fails at the point of
    actually trying to use OpenAI, not at import time.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Provider selection ───────────────────────────────────────────────────────
# Which LLM backend the gateway factory builds: "openai" or "ollama". Everything
# above the gateway depends only on the LLMGateway interface, so flipping this one
# switch repoints the whole system at a different provider — no component changes.
LLM_PROVIDER: str = os.environ.get("LLM_PROVIDER", "openai").strip().lower()

# ── OpenAI (and OpenAI-compatible providers: Groq / OpenRouter / Cerebras / Gemini) ──
OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")

# A base-URL override for the OpenAI-compatible gateway. When LLM_PROVIDER names a
# known compatible provider (groq/openrouter/cerebras/gemini), the factory sets the
# right base URL and reads that provider's key automatically; this is only needed
# for a custom/self-hosted OpenAI-compatible endpoint. Empty = real OpenAI.
LLM_BASE_URL: str = os.environ.get("LLM_BASE_URL", "")

# OpenAI-compatible providers. Set the matching *_API_KEY and select the provider
# with LLM_PROVIDER (for example, LLM_PROVIDER=groq).
GROQ_API_KEY: str = os.environ.get("GROQ_API_KEY", "")
OPENROUTER_API_KEY: str = os.environ.get("OPENROUTER_API_KEY", "")
CEREBRAS_API_KEY: str = os.environ.get("CEREBRAS_API_KEY", "")
GEMINI_API_KEY: str = os.environ.get("GEMINI_API_KEY", "")
DEEPSEEK_API_KEY: str = os.environ.get("DEEPSEEK_API_KEY", "")

# Concrete model names behind the two tiers (see ModelTier). Overridable via env
# so swapping models — or pointing at different deployments — is a config change.
MODEL_FAST: str = os.environ.get("MODEL_FAST", "gpt-4o-mini")        # cheap turns
MODEL_REASONING: str = os.environ.get("MODEL_REASONING", "gpt-4o")  # hard reasoning

# ── Ollama (local models) ────────────────────────────────────────────────────
# Used only when LLM_PROVIDER=ollama. The gateway talks to the local Ollama server
# over HTTP and needs no API key. Model names left EMPTY are AUTO-DETECTED from the
# installed models (`ollama list`): the smallest installed model backs the FAST tier
# (high-frequency guard/classifier turns), the largest backs REASONING (diagnosis).
# Pin a specific model by setting these, e.g. OLLAMA_MODEL_REASONING=qwen3:14b.
OLLAMA_HOST: str = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL_FAST: str = os.environ.get("OLLAMA_MODEL_FAST", "")
OLLAMA_MODEL_REASONING: str = os.environ.get("OLLAMA_MODEL_REASONING", "")

# ── LLM call behaviour ───────────────────────────────────────────────────────
LLM_TIMEOUT_S: float = float(os.environ.get("LLM_TIMEOUT_S", "30"))
LLM_MAX_RETRIES: int = int(os.environ.get("LLM_MAX_RETRIES", "2"))   # retries after the first try
LLM_TEMPERATURE: float = float(os.environ.get("LLM_TEMPERATURE", "0.2"))

# Per-node timeout (seconds) for the milestone's parallel executor. 20s suits a
# fast hosted model; raise it for slower local backends (Ollama on CPU/GPU), where a
# single reasoning call can take far longer and would otherwise time out and degrade
# the milestone to a "see a clinician" fallback. Override via MILESTONE_NODE_TIMEOUT_S.
MILESTONE_NODE_TIMEOUT_S: float = float(os.environ.get("MILESTONE_NODE_TIMEOUT_S", "20"))

# ── Persistence ──────────────────────────────────────────────────────────────
# SQLite database file the API opens on startup (one shared connection). Override
# via env, e.g. an absolute path in production or ":memory:" for throwaway runs.
DB_PATH: str = os.environ.get("DB_PATH", "medical_ai.db")

# ── Safety ───────────────────────────────────────────────────────────────────
# Bypasses the safety layer when false. Useful for evaluating the diagnostic
# path without being interrupted. MUST be true in production.
ENABLE_SAFETY_GUARD: bool = os.environ.get("ENABLE_SAFETY_GUARD", "true").lower() == "true"
SAFETY_LLM_TIMEOUT_S: float = float(os.environ.get("SAFETY_LLM_TIMEOUT_S", "8"))

# ── NCBI / PubMed ────────────────────────────────────────────────────────────
NCBI_API_KEY: str = os.environ.get("NCBI_API_KEY", "")
NCBI_EMAIL: str = os.environ.get("NCBI_EMAIL", "test@example.com")
NCBI_TOOL: str = os.environ.get("NCBI_TOOL", "medical_robot")
NCBI_EUTILS_BASE: str = os.environ.get("NCBI_EUTILS_BASE", "https://eutils.ncbi.nlm.nih.gov/entrez/eutils")
PUBMED_TIMEOUT_S: float = float(os.environ.get("PUBMED_TIMEOUT_S", "10.0"))
PUBMED_MAX_RESULTS: int = int(os.environ.get("PUBMED_MAX_RESULTS", "5"))

# ── Advanced RAG ─────────────────────────────────────────────────────────────
# The four Chapter-2 extension paths are production features, not figure-only
# concepts.  They are bounded so controller/search fan-out cannot grow without
# limit, and they share one PubMed client so NCBI rate limiting remains global.
ENABLE_ADVANCED_RAG: bool = (
    os.environ.get("ENABLE_ADVANCED_RAG", "true").lower() == "true"
)
RAG_FUSION_QUERY_COUNT: int = int(os.environ.get("RAG_FUSION_QUERY_COUNT", "3"))
METARAG_MAX_REPAIR_QUERIES: int = int(
    os.environ.get("METARAG_MAX_REPAIR_QUERIES", "1")
)
IMEDRAG_MAX_ITERATIONS: int = int(os.environ.get("IMEDRAG_MAX_ITERATIONS", "2"))
ADVANCED_RAG_NODE_TIMEOUT_S: float = float(
    os.environ.get("ADVANCED_RAG_NODE_TIMEOUT_S", "60")
)
ADVANCED_RAG_MAX_EVIDENCE_ITEMS: int = int(
    os.environ.get("ADVANCED_RAG_MAX_EVIDENCE_ITEMS", "12")
)

# ── Clinician-governed dynamic knowledge ─────────────────────────────────────
# The diagnostic model and the knowledge curator are intentionally configured
# independently. The curator is slower/more capable and runs only in the
# background after a visit completes; it never adds latency to patient turns.
ENABLE_DYNAMIC_KB_RAG: bool = os.environ.get("ENABLE_DYNAMIC_KB_RAG", "true").lower() == "true"
KB_SEED_DIR: str = os.environ.get("KB_SEED_DIR", "data/kb100/diseases")
KB_PENDING_DIR: str = os.environ.get("KB_PENDING_DIR", "data/kb100/pending")
KB_APPROVED_DIR: str = os.environ.get("KB_APPROVED_DIR", "data/kb100/approved_dynamic")
DYNAMIC_KB_TOPN: int = int(os.environ.get("DYNAMIC_KB_TOPN", "3"))
DYNAMIC_KB_GATE: float = float(os.environ.get("DYNAMIC_KB_GATE", "0.35"))

ENABLE_KNOWLEDGE_AGENT: bool = os.environ.get("ENABLE_KNOWLEDGE_AGENT", "true").lower() == "true"
KNOWLEDGE_AGENT_PROVIDER: str = os.environ.get("KNOWLEDGE_AGENT_PROVIDER", "deepseek").strip().lower()
KNOWLEDGE_AGENT_MODEL: str = os.environ.get("KNOWLEDGE_AGENT_MODEL", "deepseek-v4-pro").strip()
KNOWLEDGE_AGENT_FAST_MODEL: str = os.environ.get(
    "KNOWLEDGE_AGENT_FAST_MODEL", "deepseek-v4-flash"
).strip()
KNOWLEDGE_AGENT_MIN_CONFIDENCE: float = float(
    os.environ.get("KNOWLEDGE_AGENT_MIN_CONFIDENCE", "0.35")
)
KNOWLEDGE_AGENT_MAX_SOURCES: int = int(os.environ.get("KNOWLEDGE_AGENT_MAX_SOURCES", "5"))
KNOWLEDGE_AGENT_TIMEOUT_S: float = float(os.environ.get("KNOWLEDGE_AGENT_TIMEOUT_S", "180"))
KNOWLEDGE_AGENT_MAX_RETRIES: int = int(os.environ.get("KNOWLEDGE_AGENT_MAX_RETRIES", "1"))
KNOWLEDGE_AGENT_REPAIR_ATTEMPTS: int = int(
    os.environ.get("KNOWLEDGE_AGENT_REPAIR_ATTEMPTS", "2")
)
KNOWLEDGE_GUIDELINE_CATALOG: str = os.environ.get(
    "KNOWLEDGE_GUIDELINE_CATALOG", "data/kb100/_official_guideline_catalog.json"
)
OPENFDA_LABEL_BASE: str = os.environ.get(
    "OPENFDA_LABEL_BASE", "https://api.fda.gov/drug/label.json"
)
OPENFDA_API_KEY: str = os.environ.get("OPENFDA_API_KEY", "")
OPENFDA_TIMEOUT_S: float = float(os.environ.get("OPENFDA_TIMEOUT_S", "15"))
