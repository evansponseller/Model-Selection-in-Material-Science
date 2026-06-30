"""Central configuration for the Metalearning_AlloyDesign pipeline."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

INPUT_JSON = DATA_DIR / "paper_extractions.json"
OUTPUT_JSON = DATA_DIR / "paper_extractions.json"   # alias used by retrieve_data.py
OUTPUT_CSV = RESULTS_DIR / "extracted_results.csv"
# Papers the extractor flags recommend_exclude=yes go here, not the main results
EXCLUDED_CSV = RESULTS_DIR / "excluded_results.csv"

# ── Elsevier / Scopus ──────────────────────────────────────────────────────
ELSEVIER_API_KEY = os.environ.get("ELSEVIER_API_KEY", "")

SCOPUS_QUERY = (
    'TITLE-ABS-KEY('
    '("machine learning" OR "deep learning" OR "neural network" OR "gaussian process" '
    'OR "random forest" OR "support vector" OR "interatomic potential" '
    'OR "MACE" OR "MLIP" OR "moment tensor potential" OR "neuroevolution potential") '
    'AND ("alloy" OR "high-entropy alloy" OR "refractory alloy" OR "interatomic potential" '
    'OR "HEA" OR "multi-principal" OR "superalloy") '
    ') '
    'AND SRCTITLE('
    # Elsevier-only journals (full text fetchable via Elsevier Article API)
    '"Acta Materialia" OR '
    '"Computational Materials Science" OR '
    '"Materials Science and Engineering A" OR '
    '"Materials and Design" OR '
    '"Computational Materials Today" OR '
    '"Engineering Applications of Artificial Intelligence"'
    ') '
    'AND PUBYEAR > 2016'
)

SCOPUS_URL = "https://api.elsevier.com/content/search/scopus"

# Fetch by DOI
ARTICLE_URL = "https://api.elsevier.com/content/article/doi/{doi}"
# Fetch by PII (ScienceDirect internal ID visible in article URLs)
ARTICLE_PII_URL = "https://api.elsevier.com/content/article/pii/{pii}"

MAX_SCOPUS_RESULTS = 1000  # Scopus paginates in pages of 25

# ── JHU WSE AI Gateway / LLM ──────────────────────────────────────────────
AI_GATEWAY_API_KEY = os.environ.get("JHU_AI_GATEWAY_API_KEY", "")
# Compat route — OpenAI chat-completions shape, provider-prefixed model names
AI_GATEWAY_URL = "https://gateway.engineering.jhu.edu/gateway/compat/chat/completions"

# Used for classification (retrieve_data.py)
CLASSIFY_MODEL = "anthropic/claude-opus-4-8"
# Used for field extraction (extract_data.py) — higher accuracy
EXTRACT_MODEL = "anthropic/claude-opus-4-8"

HTTP_REFERER = "https://github.com/Metalearning_AlloyDesign"

# ── Claude API via JHU gateway (native Anthropic route) ───────────────────
ANTHROPIC_BASE_URL = "https://gateway.engineering.jhu.edu/gateway/anthropic"
CLAUDE_CLASSIFY_MODEL = "claude-haiku-4-5-20251001"
CLAUDE_EXTRACT_MODEL = "claude-opus-4-8"

# ── arXiv ──────────────────────────────────────────────────────────────────
ARXIV_QUERY = (
    "machine learning alloy OR high-entropy alloy OR interatomic potential "
    "OR machine learning interatomic potential OR neuroevolution potential "
    "OR moment tensor potential OR MACE potential OR random forest alloy "
    "OR gaussian process alloy OR neural network alloy"
)
ARXIV_CATEGORIES = ["cond-mat.mtrl-sci"]
ARXIV_MAX_RESULTS = 500
AR5IV_URL = "https://ar5iv.labs.arxiv.org/html/{arxiv_id}"

# ── Springer Nature (npj Computational Materials, fully open-access) ────────
# Key from https://dev.springernature.com (free tier). The OpenAccess API
# returns full-text JATS XML for OA articles.
SPRINGER_API_KEY = os.environ.get("SPRINGER_API_KEY", "")
SPRINGER_META_URL = "https://api.springernature.com/openaccess/json"
SPRINGER_JATS_URL = "https://api.springernature.com/openaccess/jats"
# npj Computational Materials — eISSN 2057-3960
SPRINGER_ISSN = "2057-3960"
SPRINGER_QUERY = (
    '("machine learning" OR "deep learning" OR "neural network" '
    'OR "gaussian process" OR "random forest" OR "support vector" '
    'OR "interatomic potential" OR "MLIP") '
    'AND (alloy OR "high-entropy alloy" OR "refractory alloy" OR HEA '
    'OR "multi-principal" OR superalloy) '
    f'AND issn:{SPRINGER_ISSN} AND datefrom:2017-01-01'
)
SPRINGER_MAX_RESULTS = 500

# ── Extraction settings ────────────────────────────────────────────────────
# Max characters of retrieved context sent to LLM per field question
MAX_CONTEXT_CHARS = 3000
# Number of sentences from the end of the introduction used for classification
CLASSIFY_INTRO_SENTENCES = 12
# Rate-limit pause between API calls (seconds)
RATE_LIMIT_PAUSE = 1.0
# Seconds to wait after a 429 response before retrying
RATE_LIMIT_RETRY_PAUSE = 30
# Max retries on transient network/rate-limit errors
MAX_RETRIES = 5

# ── Section skip-list (cite others' methods; not this paper's) ────────────
BOILERPLATE_SECTIONS = {
    "abstract",
    "introduction",
    "related work",
    "acknowledgements",
    "acknowledgments",
    "references",
    "supplementary",
    "appendix",
    "funding",
    "declaration of competing interest",
    "credit authorship contribution statement",
}

# Sections skipped during field extraction. Unlike BOILERPLATE_SECTIONS, this
# KEEPS supplementary/appendix — those are the authors' own extended methods,
# where feature lists and dataset details often live (reviewer-flagged). Only
# sections describing others' work or front/back matter are dropped.
EXTRACT_SKIP_SECTIONS = BOILERPLATE_SECTIONS - {"supplementary", "appendix"}

# ── Extraction fields ──────────────────────────────────────────────────────
EXTRACTION_FIELDS = [
    "ml_models",
    "target_properties",
    "dataset_size",
    "data_type",
    "features",
    "num_features",
    "performance_metric",
]

# Keywords used for RAG sentence scoring per field.
# Design notes:
#   ml_models    — removed generic "model"/"algorithm" (too noisy); added author-action phrases
#                  and specific architectures seen in the corpus (CORAL, NEP, PACE, etc.)
#   target_props — removed bare "energy"/"force" (pulls MLIP training targets instead of the
#                  scientific goal); kept compound terms and added property-domain keywords
#   dataset_size — added total/collected/obtained phrases that anchor the count sentence
FIELD_KEYWORDS: dict[str, list[str]] = {
    "ml_models": [
        # Author-action phrases (high precision — these sentences describe the authors' own work)
        "we trained", "we developed", "we fitted", "we employed", "we used",
        "we propose", "we constructed", "was trained", "were trained", "is trained",
        # Specific model families in the corpus
        "random forest", "gradient boosting", "xgboost", "gaussian process", "gpr",
        "support vector", "svr", "svm", "neural network", "deep neural", "dnn",
        "convolutional", "cnn", "lstm", "attention", "transformer", "bert",
        "linear regression", "ridge regression", "lasso", "symbolic regression",
        "kriging", "surrogate",
        # Tree-based families (reviewer-flagged: Regression Trees were missed)
        "regression tree", "decision tree", "extra trees", "extremely randomized",
        "lightgbm", "catboost", "adaboost", "ensemble", "bagging",
        # Interatomic potential families (MLIPs)
        "mlip", "mace", "mtp", "nnp", "nep", "ace", "pace", "mlff",
        "moment tensor", "neuroevolution", "atomic cluster expansion",
        "machine learning potential", "machine learning force field",
        # Other specific names seen in the corpus
        "coral", "cluster expansion", "bayesian",
    ],
    "target_properties": [
        # Author-intent phrases
        "we aim to predict", "our model predicts", "the model predicts",
        "trained to predict", "used to predict", "goal of this", "objective",
        "the target", "output variable", "target variable", "label",
        # Scientific property terms (compound first to score higher than substrings)
        "thermal conductivity", "electrical conductivity",
        "yield strength", "ultimate tensile strength", "tensile strength",
        "formation energy", "mixing enthalpy", "binding energy",
        "liquidus temperature", "melting point", "transformation temperature",
        "short-range order", "sro", "bandgap", "band gap",
        "corrosion", "hardness", "elongation", "ductility",
        "lattice parameter", "elastic constant", "bulk modulus", "shear modulus",
        "diffusion", "vibrational", "superelastic", "hysteresis",
        # Generic but still useful
        "predict", "property", "target property",
    ],
    "dataset_size": [
        # Anchor phrases that appear near the count
        "total of", "consisting of", "containing", "a dataset of",
        "we collected", "were collected", "were obtained", "were calculated",
        "we generated", "were generated", "dft calculations", "dft-calculated",
        "training set", "training data", "data points", "data set",
        "samples", "configurations", "structures", "instances",
        "observations", "entries", "experiments",
    ],
    "data_type": [
        # Anchor on the SOURCE of the training data, not any method mentioned
        "training data", "training set", "training dataset", "dataset was",
        "data were collected", "data were generated", "data were obtained",
        "data were calculated", "data were measured", "data originate",
        "compiled from", "collected from", "obtained from",
        # Experimental indicators
        "experimental", "experiment", "measured", "laboratory", "fabricated",
        # Computational indicators
        "dft", "density functional", "molecular dynamics", "md simulation",
        "calphad", "ab initio", "first-principles", "first principles",
        "computational", "calculated", "simulated", "synthetic",
    ],
    "features": [
        # Anchor phrases
        "input feature", "input descriptor", "feature vector", "descriptor vector",
        "we use", "we used", "features include", "descriptors include",
        "feature set", "descriptor set", "feature space",
        # Feature type terms
        "composition", "elemental", "atomic number", "atomic radius",
        "electronegativity", "valence", "thermodynamic", "structural",
        "fingerprint", "representation", "embedding",
        # Generic fallbacks
        "feature", "descriptor",
    ],
    "num_features": [
        "number of features", "number of descriptors", "number of inputs",
        "total features", "total descriptors", "input variables",
        "dimensionality", "feature dimension",
        "features were", "descriptors were", "feature", "descriptor",
    ],
    "performance_metric": [
        # Common metric names
        "r²", "r2", "r-squared", "coefficient of determination",
        "rmse", "root mean square", "mean absolute error", "mae",
        "mean squared error", "mse",
        "mean absolute percentage error", "mape",
        "accuracy", "precision", "recall", "f1 score", "f1-score", "auc",
        # Phrases that anchor a reported value
        "achieved", "obtained", "yielded", "reported",
        "test error", "test rmse", "test mae", "test r2",
        "training error", "validation error",
        "cross-validation", "cross validation",
        "prediction error", "generalization",
        "outperform", "better than", "compared to",
    ],
}
