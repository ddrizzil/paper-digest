"""
paper_digest_service.py
Daily Paper Digest with NLP-enhanced scoring, citation-aware ranking,
learning-based personalization, trending detection, and HTML email digest.
"""
import csv
import logging
import math
import os
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from itertools import islice
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from urllib.parse import quote_plus

import feedparser
import numpy as np
import requests
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Attempt to import scholarly + proxy support; continue gracefully if absent.
try:
    from scholarly import ProxyGenerator, scholarly  # type: ignore
except ImportError:
    scholarly = None
    ProxyGenerator = None

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
except ImportError:
    SentenceTransformer = None

# ---------------- CONFIG ----------------
SENDER = "dailypapersender@intrawebb.com"
DEFAULT_RECIPIENT = "daniel@intrawebb.com"

FEEDS = [
    # ============ AREAS: CULTURAL HERITAGE & IMAGING ============
    "https://heritagesciencejournal.springeropen.com/articles/rss.xml",
    "https://www.sciencedirect.com/journal/journal-of-cultural-heritage/rss",
    "https://www.osapublishing.org/ao/rss.cfm",
    "https://www.osapublishing.org/oe/rss.cfm",
    "https://www.osapublishing.org/boe/rss.cfm",
    "https://www.osapublishing.org/josaa/rss.cfm",
    "https://www.osapublishing.org/josab/rss.cfm",
    "https://www.osapublishing.org/ol/rss.cfm",
    "https://www.osapublishing.org/optica/rss.cfm",
    "https://www.nature.com/lsa.rss",
    "https://www.spiedigitallibrary.org/journals/journal-of-electronic-imaging/rss",
    "https://www.sciencedirect.com/journal/optics-and-lasers-in-engineering/rss",
    "https://www.sciencedirect.com/journal/optics-communications/rss",
    "https://www.sciencedirect.com/journal/studies-in-conservation/rss",
    "https://www.mdpi.com/journal/heritage/rss",
    "https://brill.com/view/journals/ins/ins-overview.xml",
    "https://www.mdpi.com/journal/arts/rss",
    "https://journals.sagepub.com/action/showFeed?ui=0&mi=ehikzz&ai=2b4&jc=dsaa&type=etoc&feed=rss",

    # ============ RF SYSTEMS, MICROWAVE, RADAR ============
    "https://ieeexplore.ieee.org/rss/TOC78.XML",
    "https://ieeexplore.ieee.org/rss/TOC5.XML",
    "https://ieeexplore.ieee.org/rss/TOC7361.XML",
    "https://ieeexplore.ieee.org/rss/TOC10376.XML",
    "https://ieeexplore.ieee.org/rss/TOC22.XML",
    "https://ieeexplore.ieee.org/rss/TOC8.XML",
    "https://ieeexplore.ieee.org/rss/TOC4234.XML",
    "https://ieeexplore.ieee.org/rss/TOC6668.XML",
    "https://ieeexplore.ieee.org/rss/TOC87.XML",
    "https://ieeexplore.ieee.org/rss/TOC8859.XML",
    "https://ieeexplore.ieee.org/rss/TOC36.XML",
    "https://ieeexplore.ieee.org/rss/TOC4609.XML",

    # ============ SIGNAL PROCESSING & MACHINE LEARNING ============
    "https://ieeexplore.ieee.org/rss/TOC97.XML",
    "https://ieeexplore.ieee.org/rss/TOC4200.XML",
    "https://ieeexplore.ieee.org/rss/TOC6046.XML",
    "https://ieeexplore.ieee.org/rss/TOC6287.XML",
    "https://ieeexplore.ieee.org/rss/TOC34.XML",
    "https://ieeexplore.ieee.org/rss/TOC6221.XML",
    "https://www.sciencedirect.com/journal/signal-processing/rss",
    "https://www.sciencedirect.com/journal/digital-signal-processing/rss",

    # ============ CLIMATE, ENVIRONMENT, AGRICULTURE ============
    "https://www.nature.com/subjects/remote-sensing/rss",
    "https://www.nature.com/nclimate.rss",
    "https://www.nature.com/natrevearth.rss",
    "https://www.mdpi.com/journal/atmosphere/rss",
    "https://www.mdpi.com/journal/climate/rss",
    "https://journals.ametsoc.org/rss/bams.xml",
    "https://agupubs.onlinelibrary.wiley.com/feed/19448007/most-recent",
  #3  "https://www.sciencedirect.com/journal/agricultural-and-forest-meteorology/rss",
    "https://www.mdpi.com/journal/land/rss",
    "https://www.mdpi.com/journal/sustainability/rss",
    "https://nhess.copernicus.org/xml/rss2_0.xml",
    "https://www.mdpi.com/journal/geosciences/rss",

    # ============ MATERIALS SCIENCE & SPECTROSCOPY ============
    "https://www.mdpi.com/journal/photonics/rss",
    "https://www.nature.com/nphoton.rss",
    "https://www.sciencedirect.com/journal/spectrochimica-acta-part-a-molecular-and-biomolecular-spectroscopy/rss",
    "https://pubs.rsc.org/en/journals/rss/ja",
    "https://www.mdpi.com/journal/materials/rss",
    # "https://pubs.acs.org/action/showFeed?type=axatoc&feed=rss&jc=ancham",

    # ============ APPLIED PHYSICS & INSTRUMENTATION ============
    "https://aip.scitation.org/rss/content/aip/journal/rsi?sc_cid=rss",
    "https://www.nature.com/scientificreports.rss",
    "https://iopscience.iop.org/journal/rss/0957-0233",
    "https://www.mdpi.com/journal/applsci/rss",
    "https://www.mdpi.com/journal/instruments/rss",

    # ============ WIRELESS COMMUNICATIONS & IOT ============
    "https://ieeexplore.ieee.org/rss/TOC26.XML",
    "https://ieeexplore.ieee.org/rss/TOC35.XML",
    "https://ieeexplore.ieee.org/rss/TOC6570650.XML",
    "https://ieeexplore.ieee.org/rss/TOC9739572.XML",

    # ============ COMPUTER VISION & IMAGING ============
    "https://link.springer.com/search.rss?facet-content-type=Article&facet-journal-id=11263&channel-name=International+Journal+of+Computer+Vision",
    "https://www.mdpi.com/journal/jimaging/rss",

    # ============ MULTIDISCIPLINARY & HIGH-IMPACT ============
    "https://www.science.org/rss/news_current.xml",
    "https://www.nature.com/nature.rss",
    "https://www.pnas.org/rss/latest.xml",

    # ============ ARXIV FEEDS ============
    "https://arxiv.org/rss/eess.SP",
    "https://arxiv.org/rss/cs.LG",
    "https://arxiv.org/rss/cs.CV",
    "https://arxiv.org/rss/cs.AI",
    "https://arxiv.org/rss/stat.ML",
    "https://arxiv.org/rss/physics.ao-ph",
    "https://arxiv.org/rss/physics.app-ph",
    "https://arxiv.org/rss/physics.data-an",
    "https://arxiv.org/rss/physics.geo-ph",
    "https://arxiv.org/rss/cs.RO",
    "https://arxiv.org/rss/quant-ph",
    "https://arxiv.org/rss/eess.IV",
    "https://arxiv.org/rss/physics.optics",
    "https://arxiv.org/rss/physics.ins-det",
]

# Expanded keyword groups spanning commercial radar, climate, & disaster response.
KEYWORD_GROUPS = {
    "cultural_heritage": [
        "cultural heritage", "artwork preservation", "conservation science",
        "art conservation", "heritage digitization", "archaeological remote sensing",
        "museum imaging", "collection risk assessment", "preventive conservation"
    ],
    "rf_systems_core": [
        "RF component", "microwave component", "nonlinear radar", "harmonic radar",
        "ground penetrating radar", "high-power microwave", "intermodulation distortion",
        "passive intermodulation", "rf fingerprinting", "volterra series"
    ],
    "optical_techniques": [
        "optical coherence tomography", "OCT imaging", "reflectance spectroscopy",
        "multispectral imaging", "hyperspectral imaging", "infrared reflectography",
        "UV fluorescence", "x-ray fluorescence mapping", "raman spectroscopy",
        "photoacoustic imaging", "laser ultrasound", "terahertz imaging"
    ],
    "ml_methods": [
        "deep learning", "convolutional neural network", "cnn",
        "recurrent neural network", "rnn", "lstm", "transformer model",
        "attention mechanism", "generative adversarial network", "gan",
        "variational autoencoder", "vae", "reinforcement learning",
        "transfer learning", "few-shot learning", "meta-learning",
        "self-supervised learning", "contrastive learning",
        "neural architecture search", "automl", "federated learning",
        "graph neural network", "physics-informed neural network", "pinn"
    ],
    "ml_signal_processing": [
        "deep learning signal processing", "neural network denoising",
        "learned compression", "neural beamforming", "ai-based detection",
        "learned reconstruction", "model-based deep learning", "unrolled optimization",
        "differentiable signal processing", "end-to-end learning", "neural operator",
        "learned sensing"
    ],
    "advanced_dsp": [
        "sparse representation", "dictionary learning", "matrix completion",
        "low-rank approximation", "tensor decomposition", "tensor factorization",
        "blind source separation", "independent component analysis",
        "nonnegative matrix factorization", "nmf", "bayesian inference",
        "variational inference", "expectation maximization", "particle filter",
        "kalman filter", "extended kalman filter", "unscented kalman filter",
        "ensemble kalman filter"
    ],
    "nonlinear_systems": [
        "nonlinear dynamics", "nonlinear characterization", "harmonic generation",
        "intermodulation distortion", "volterra series", "polynomial modeling",
        "behavioral modeling", "digital predistortion", "power amplifier linearization",
        "nonlinear compensation", "chaos theory", "bifurcation analysis",
        "lyapunov exponent", "strange attractor"
    ],
    "rf_diagnostics": [
        "rf diagnostics", "component health monitoring", "predictive maintenance rf",
        "anomaly detection rf", "fault detection radar", "performance degradation detection",
        "automated test equipment", "ate", "rf calibration",
        "vector network analyzer", "vna", "spectrum analyzer", "signal analyzer",
        "modulation analysis", "electromagnetic interference", "emi testing",
        "rf front-end testing", "receiver sensitivity", "transmitter linearity",
        "phase noise measurement","cryogenic"
    ],
    "5g_6g_systems": [
        "5g nr", "new radio", "6g wireless", "massive mimo", "beamforming 5g",
        "millimeter wave communication", "mmwave 5g", "terahertz communication",
        "thz wireless", "reconfigurable intelligent surface", "ris", "network slicing",
        "edge computing", "ultra-reliable low-latency", "urllc",
        "enhanced mobile broadband", "embb", "o-ran", "open ran", "virtualized ran"
    ],
    "automotive_radar": [
        "automotive radar", "adas radar", "autonomous driving radar",
        "77 ghz radar", "79 ghz radar", "fmcw radar automotive", "radar sensor fusion",
        "vehicle-to-everything", "v2x", "radar point cloud", "4d radar imaging",
        "radar target classification", "pedestrian detection radar",
        "collision avoidance radar", "blind spot detection",
        "adaptive cruise control radar", "parking assist radar"
    ],
    "satellite_systems": [
        "synthetic aperture radar", "sar", "insar", "differential insar", "dinsar",
        "satellite remote sensing", "earth observation", "sentinel", "landsat",
        "modis", "sar interferometry", "polarimetric sar", "sar tomography",
        "bistatic radar", "spaceborne radar", "satellite altimetry",
        "gnss reflectometry", "gnss-r", "cubesat", "smallsat", "nanosatellite"
    ],
    "climate_methods": [
        "climate modeling", "climate projection", "downscaling", "data assimilation",
        "reanalysis", "climate feedback", "radiative forcing", "climate sensitivity",
        "tipping point", "extreme event attribution", "seasonal forecasting",
        "subseasonal prediction", "ensemble prediction", "uncertainty quantification"
    ],

    "precision_agriculture": [
        "precision agriculture", "smart farming", "crop monitoring",
        "yield prediction", "soil sensing", "irrigation management",
        "variable rate application", "site-specific management",
        "agricultural drone", "uav agriculture", "plant phenotyping",
        "canopy sensing", "nitrogen sensing", "chlorophyll fluorescence",
        "crop disease detection", "pest monitoring sensor"
    ],
    "quantum_technologies": [
        "quantum sensing", "quantum radar", "quantum illumination", "quantum imaging",
        "quantum metrology", "quantum communication", "quantum key distribution",
        "qkd", "entanglement", "squeezed state", "single photon detector",
        "superconducting qubit", "nitrogen vacancy center", "quantum computing"
    ],
    
    # ============ PHOTOACOUSTICS (for your art imaging) ============
    "photoacoustics": [
        "photoacoustic imaging", "photoacoustic spectroscopy", "optoacoustic",
        "thermoacoustic", "laser ultrasound", "acoustic microscopy",
        "photoacoustic tomography", "depth profiling photoacoustic"
    ],
    "heritage_materials": [
        "organic binder", "egg tempera", "oil paint", "acrylic paint",
        "watercolor", "gouache", "encaustic", "fresco", "tempera grassa",
        "azurite", "malachite", "cinnabar", "orpiment", "realgar",
        "verdigris", "prussian blue", "chrome yellow", "titanium white",
        "carbon black", "ivory black", "lamp black", "natural resin",
        "dammar", "mastic resin", "shellac", "copal", "beeswax"
    ],
    "digital_humanities": [
        "digital humanities", "computational humanities", "cultural analytics",
        "distant reading", "text mining humanities", "network analysis humanities",
        "gis humanities", "spatial humanities", "3d humanities",
        "virtual reconstruction", "digital archive", "linked open data",
        "semantic web humanities", "ontology cultural heritage"
    ],
    "infrastructure_monitoring": [
        "structural health monitoring", "shm", "bridge monitoring", "building monitoring",
        "pipeline inspection", "tunnel monitoring", "dam monitoring",
        "wind turbine monitoring", "railway infrastructure", "pavement condition",
        "concrete deterioration", "corrosion monitoring", "vibration monitoring",
        "modal analysis", "non-destructive evaluation", "nde",
        "acoustic emission", "guided wave"
    ],
    "iot_wsn": [
        "internet of things", "iot", "wireless sensor network", "wsn",
        "lora", "lorawan", "nb-iot", "lte-m", "zigbee", "bluetooth low energy",
        "ble", "edge ai", "tinyml", "on-device learning", "energy harvesting",
        "battery-free sensing", "backscatter communication", "ambient backscatter",
        "rfid sensor", "nfc sensor", "fog computing", "mesh network"
    ],
    "proptech_real_estate": [
        "property technology", "proptech", "real estate analytics",
        "property valuation", "automated valuation model", "avm",
        "real estate prediction", "housing market", "gentrification modeling",
        "urban development", "real estate investment", "portfolio optimization",
        "property risk assessment", "location intelligence", "spatial economics",
        "hedonic pricing", "neighborhood analysis", "walkability index"
    ],
    "optimization_control": [
        "convex optimization", "non-convex optimization", "stochastic optimization",
        "robust optimization", "distributed optimization", "online optimization",
        "optimal control", "model predictive control", "mpc", "adaptive control",
        "sliding mode control", "h-infinity control", "lqr", "lqg",
        "game theory", "nash equilibrium", "multi-agent system",
        "consensus algorithm"
    ],
    "inverse_problems": [
        "inverse problem", "ill-posed problem", "regularization",
        "tikhonov regularization", "total variation", "tv regularization",
        "iterative reconstruction", "algebraic reconstruction",
        "maximum likelihood estimation", "maximum a posteriori",
        "bayesian inversion", "compressed sensing reconstruction",
        "image deconvolution", "blind deconvolution", "super-resolution",
        "phase retrieval"
    ],
    "time_series": [
        "time series analysis", "time series forecasting", "arima",
        "autoregressive", "state space model", "kalman smoothing",
        "trend analysis", "seasonality", "changepoint detection",
        "anomaly detection time series", "outlier detection", "recurrent neural network time series",
        "lstm forecasting", "temporal convolutional network", "tcn",
        "attention-based time series"
    ],
    "optics_fundamentals": [
        "optical fiber", "fiber optic", "waveguide", "optical sensor",
        "photonic sensor", "optical filter", "dichroic", "bandpass filter",
        "optical coherence", "interferometry", "michelson interferometer",
        "fabry-perot", "diffraction", "scattering", "absorption",
        "fluorescence", "phosphorescence", "luminescence",
        "polarization", "birefringence", "dichroism"
    ],
    "sustainability_esg": [
        "sustainability monitoring", "esg reporting", "carbon footprint",
        "greenhouse gas emission", "circular economy", "life cycle assessment",
        "environmental impact", "biodiversity net gain", "nature-based solution",
        "ecosystem service", "climate adaptation", "climate resilience",
        "renewable energy", "solar", "wind energy", "energy efficiency",
        "smart grid", "sustainable agriculture", "regenerative agriculture"
    ],
}

RESEARCH_PROFILE = (
    "I research nonlinear electromagnetic phenomena in RF systems, using machine learning "
    "for component identification and damage detection in high-power RF components. "
    "I also work on multimodal optical imaging for art conservation, including OCT, "
    "hyperspectral imaging, XRF, and Raman spectroscopy for pigment analysis and brushstroke characterization. "
    "I'm interested in commercial applications of RF diagnostics in 5G/6G infrastructure and automotive radar, "
    "as well as climate monitoring using radar remote sensing."
)

SEMANTIC_MODEL = None
RESEARCH_EMBEDDING = None
if SentenceTransformer is not None:
    try:
        SEMANTIC_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        RESEARCH_EMBEDDING = SEMANTIC_MODEL.encode(RESEARCH_PROFILE, convert_to_numpy=True)
    except Exception as exc:
        logging.getLogger(__name__).warning("Failed to load sentence transformer model: %s", exc)
        SEMANTIC_MODEL = None
        RESEARCH_EMBEDDING = None

ALL_KEYWORDS = [phrase for phrases in KEYWORD_GROUPS.values() for phrase in phrases]
KEYWORD_TO_GROUP: Dict[str, str] = {}
for group, phrases in KEYWORD_GROUPS.items():
    for phrase in phrases:
        KEYWORD_TO_GROUP[phrase.lower()] = group

GROUP_WEIGHT_RF = 1.5
GROUP_WEIGHT_HERITAGE = 1.3
GROUP_WEIGHT_DEFAULT = 1.0

DOI_PATTERN = re.compile(r"^10\.\d{4,9}/\S+$", re.IGNORECASE)

ADJACENT_KEYWORDS = [
    "quantum radar", "cognitive radio", "digital twin environment",
    "satellite-ground fusion", "ESG reporting technology",
    "environmental signal processing", "climate resilience analytics",
    "sustainable infrastructure monitoring", "smart grid sensing"
]

LOG_FILE = "logs/paper_digest_log.csv"
PERSONALIZATION_FILE = "config/clicks.txt"
os.makedirs("logs", exist_ok=True)
os.makedirs("config", exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Daily vs weekly digest mode toggle.
DIGEST_MODE = os.getenv("DIGEST_MODE", "daily").lower()

# Backoff / rate limit configuration.
SCHOLAR_RESULTS_PER_KEYWORD = 3
SCHOLAR_RATE_LIMIT_SECONDS = 2.5
SCHOLAR_MAX_KEYWORDS = 8
SCHOLAR_MAX_FAILURES = 3

CROSSREF_ENDPOINT = "https://api.crossref.org/works"
CROSSREF_ROWS_PER_KEYWORD = 20
CROSSREF_TIMEOUT = 10
CROSSREF_MAX_KEYWORDS = 10

ADJACENT_TARGET = 10
RECENT_TARGET = 10
DECADE_TARGET = 10
DECADE_MIN_SCORE = 6.0

ML_RF_GROUPS = {
    "rf_systems_core",
    "rf_diagnostics",
    "nonlinear_systems",
    "ml_methods",
    "ml_signal_processing",
    "advanced_dsp",
    "5g_6g_systems",
    "automotive_radar",
    "satellite_systems",
    "optimization_control",
    "inverse_problems",
    "time_series",
    "iot_wsn",
}

HERITAGE_OPTICS_GROUPS = {
    "cultural_heritage",
    "optical_techniques",
    "heritage_materials",
    "photoacoustics",
    "digital_humanities",
    "optics_fundamentals",
}

PRIORITY_ML_KEYWORDS = {
    "harmonic radar",
    "nonlinear radar",
    "harmonic generation",
    "volterra series",
    "rf diagnostics",
    "component health monitoring",
    "passive intermodulation",
}

CITATION_CACHE: Dict[str, int] = {}
SESSION = requests.Session()

# ---------------- HELPERS ----------------
def configure_scholarly_proxy() -> None:
    """Configure scholarly to use optional proxy from SCHOLAR_PROXY_URL."""
    if scholarly is None or ProxyGenerator is None:
        return
    proxy_url = os.getenv("SCHOLAR_PROXY_URL")
    if not proxy_url:
        return
    pg = ProxyGenerator()
    try:
        if proxy_url.startswith("socks5://"):
            parsed = re.sub(r"^socks5://", "", proxy_url)
            creds_host = parsed.split("@")
            if len(creds_host) == 2:
                creds, host_port = creds_host
                user, password = creds.split(":", 1)
            else:
                user = password = None
                host_port = creds_host[0]
            host, port = host_port.split(":", 1)
            success = pg.SOCKS5(
                proxy_host=host,
                proxy_port=int(port),
                username=user,
                password=password,
            )
        else:
            success = pg.SingleProxy(http=proxy_url, https=proxy_url)
        if success:
            scholarly.use_proxy(pg)
            logger.info("Configured scholarly proxy via SCHOLAR_PROXY_URL.")
        else:
            logger.warning("Failed to configure scholarly proxy; continuing without proxy.")
    except Exception as err:
        logger.warning("Error configuring scholarly proxy: %s", err)

configure_scholarly_proxy()

def ensure_file(path: str, default: str = "") -> None:
    """Create a file with default contents if it does not exist."""
    file_path = Path(path)
    if not file_path.exists():
        file_path.write_text(default, encoding="utf-8")

def request_with_backoff(url: str, *, params=None, headers=None, method: str = "GET",
                         max_attempts: int = 5, base_delay: float = 1.0):
    """HTTP helper with exponential backoff for rate-limited APIs."""
    delay = base_delay
    for attempt in range(1, max_attempts + 1):
        try:
            response = SESSION.request(method, url, params=params, headers=headers,
                                       timeout=CROSSREF_TIMEOUT)
            if response.status_code == 429 or response.status_code >= 500:
                logger.info("Rate limit (%s) on %s attempt %d/%d. Sleeping %.1fs.",
                            response.status_code, url, attempt, max_attempts, delay)
                time.sleep(delay)
                delay *= 2
                continue
            response.raise_for_status()
            return response
        except requests.RequestException as err:
            if attempt == max_attempts:
                raise
            logger.warning("Request error %s on %s attempt %d/%d. Sleeping %.1fs.",
                           err, url, attempt, max_attempts, delay)
            time.sleep(delay)
            delay *= 2
    return None


def sanitize_doi(raw: str) -> str:
    """Return a bare DOI string if the input looks like a DOI; otherwise empty."""
    if not raw:
        return ""
    candidate = raw.strip()
    candidate = candidate.replace("https://doi.org/", "").replace("http://doi.org/", "")
    candidate = re.sub(r"^doi:\s*", "", candidate, flags=re.IGNORECASE)
    candidate = candidate.strip()
    if DOI_PATTERN.match(candidate):
        return candidate
    return ""


def ensure_summary_text(paper: dict) -> None:
    """Provide a fallback summary when metadata lacks an abstract."""
    summary = (paper.get("summary") or "").strip()
    if summary:
        paper["summary"] = summary
        return
    title = paper.get("title", "This paper") or "This paper"
    published = paper.get("published")
    year = published.year if isinstance(published, datetime) else ""
    paper["summary"] = f"Summary unavailable. Refer to the full text for details. {year}".strip()

def load_recipients() -> List[str]:
    # Primary: Check for EMAIL_RECIPIENTS GitHub Secret
    env_recipients = os.getenv("EMAIL_RECIPIENTS", "").strip()
    if env_recipients:
        # Can be comma-separated or newline-separated
        recipients = [e.strip() for e in env_recipients.replace(",", "\n").split("\n") if e.strip()]
        if recipients:
            logger.info("Loaded %d recipient(s) from EMAIL_RECIPIENTS secret: %s", len(recipients), ", ".join(recipients))
            return recipients
        else:
            logger.warning("EMAIL_RECIPIENTS secret is set but empty after parsing")
    
    # Fallback: Read from file (for local testing or if secret not set)
    path = "config/emails.txt"
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            recipients = [e.strip() for e in f if e.strip()]
            if recipients:
                logger.info("Loaded %d recipient(s) from %s file: %s", len(recipients), path, ", ".join(recipients))
                return recipients
            else:
                logger.warning("emails.txt file exists but is empty")
    else:
        logger.info("emails.txt file not found at %s", path)
    
    # Final fallback: Use default recipient
    logger.warning("No EMAIL_RECIPIENTS secret or emails.txt file found, using default recipient: %s", DEFAULT_RECIPIENT)
    return [DEFAULT_RECIPIENT]

def load_click_history() -> List[str]:
    """Load clicked paper titles for personalization."""
    ensure_file(PERSONALIZATION_FILE)
    with open(PERSONALIZATION_FILE, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def update_keyword_weights(clicked_titles: Iterable[str]) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Learn keyword weights from click history.
    Each phrase match yields +0.1 weight; returns weight dict and normalized stats.
    """
    weights: Dict[str, float] = defaultdict(float)
    title_texts = [title.lower() for title in clicked_titles]
    for phrase in ALL_KEYWORDS:
        phrase_l = phrase.lower()
        for title in title_texts:
            if phrase_l in title:
                weights[phrase] += 0.1
    total_weight = sum(weights.values()) or 1.0
    normalized = {phrase: round(weight / total_weight, 3) for phrase, weight in weights.items()}
    return dict(weights), normalized


def nlp_score_keyword_only(text: str, learned_weights: Dict[str, float]) -> float:
    if not text or not text.strip():
        return 0.0
    tokens = [t.lower() for t in re.findall(r"\w+", text.lower())]
    score = 0.0
    for phrase in ALL_KEYWORDS:
        phrase_tokens = [t.lower() for t in phrase.lower().split()]
        for i in range(len(tokens) - len(phrase_tokens) + 1):

            if tokens[i:i + len(phrase_tokens)] == phrase_tokens:
                group = KEYWORD_TO_GROUP.get(phrase.lower(), "")
                if group in ML_RF_GROUPS or has_priority_topic({"title": phrase, "summary": ""}):
                    base_weight = GROUP_WEIGHT_RF
                elif group in HERITAGE_OPTICS_GROUPS:
                    base_weight = GROUP_WEIGHT_HERITAGE
                else:
                    base_weight = GROUP_WEIGHT_DEFAULT
                score += base_weight * (1.0 + learned_weights.get(phrase, 0.0))
                break

    try:
        vectorizer = TfidfVectorizer()
        tfidf = vectorizer.fit_transform([text] + ALL_KEYWORDS)
        sim_score = cosine_similarity(tfidf[0:1], tfidf[1:]).max()
    except ValueError:
        sim_score = 0.0

    keyword_score = score + sim_score * 2.0
    return max(0.0, min(10.0, keyword_score))


def semantic_similarity_score(text: str) -> float:
    if not text or not text.strip() or SEMANTIC_MODEL is None or RESEARCH_EMBEDDING is None:
        return 0.0
    try:
        truncated = " ".join(text.split()[:400])
        embedding = SEMANTIC_MODEL.encode(truncated, convert_to_numpy=True)
        ref_vec = np.asarray(RESEARCH_EMBEDDING)
        if np.linalg.norm(embedding) == 0 or np.linalg.norm(ref_vec) == 0:
            return 0.0
        similarity = float(np.dot(ref_vec, embedding) / (np.linalg.norm(ref_vec) * np.linalg.norm(embedding)))
        score = similarity * 15.0
        return max(0.0, min(10.0, score))
    except Exception as exc:
        logger.warning("Semantic scoring failed: %s", exc)
        return 0.0


def nlp_score(text: str, learned_weights: Dict[str, float]) -> float:
    keyword_only = nlp_score_keyword_only(text, learned_weights)
    semantic = semantic_similarity_score(text)
    if semantic == 0.0 and (SEMANTIC_MODEL is None or RESEARCH_EMBEDDING is None):
        return keyword_only
    return max(0.0, min(10.0, 0.7 * semantic + 0.3 * keyword_only))

def enhanced_score(paper: dict, now: datetime, learned_weights: Dict[str, float]) -> float:
    """
    Combine NLP score with citation-based boost.
    Older (>2 years) highly cited (>100) papers get additional credit,
    capped at +5 to avoid overpowering recent work.
    """
    text = f"{paper.get('title', '')} {paper.get('summary', '')}"
    base_score = nlp_score(text, learned_weights)
    citations = paper.get("citations", 0) or 0
    citation_boost = 0.0
    if citations > 0:
        citation_boost = min(5.0, math.log10(citations + 1))
        paper_age_years = max(0.0, (now - paper["published"]).days / 365.25)
        if paper_age_years > 2 and citations > 100:
            extra = (citations - 100) / 200.0
            citation_boost = min(5.0, citation_boost + extra)
    return round(base_score + min(5.0, citation_boost), 3)

def short_summary(text: str, max_sent: int = 2) -> str:
    try:
        sents = re.split(r"(?<=[.!?]) +", text)
        if len(sents) <= max_sent:
            return text
        vec = TfidfVectorizer(stop_words="english")
        tfidf = vec.fit_transform(sents)
        sims = cosine_similarity(tfidf[-1:], tfidf).flatten()
        top = sims.argsort()[-max_sent:][::-1]
        return " ".join([sents[i] for i in sorted(top)])
    except Exception:
        return (text or "")[:300] + "..."

def abstract_preview(text: str, sentences: int = 3) -> str:
    """Return the first few sentences of an abstract for high-relevance highlights."""
    if not text:
        return ""
    sents = re.split(r"(?<=[.!?]) +", text)
    return " ".join(sents[:sentences])

def parse_feed(url: str):
    try:
        return feedparser.parse(url).entries
    except Exception as e:
        logger.error("Error parsing %s: %s", url, e)
        return []

def extract_pub_date(entry) -> datetime:
    dt_struct = entry.get("published_parsed") or entry.get("updated_parsed")
    if dt_struct:
        try:
            timestamp = feedparser.mktime_tz(dt_struct)
            return datetime.fromtimestamp(timestamp, tz=timezone.utc)
        except (TypeError, OverflowError, AttributeError):
            pass
    for key in ("published", "updated"):
        val = entry.get(key)
        if not val:
            continue
        try:
            dt = parsedate_to_datetime(val)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            return dt
        except (TypeError, ValueError):
            continue
    return None

def extract_crossref_date(item: dict) -> datetime:
    for key in ("published-print", "published-online", "issued", "created", "deposited"):
        date_info = item.get(key)
        if not date_info:
            continue
        parts = date_info.get("date-parts")
        if not parts:
            continue
        date_parts = parts[0]
        if not date_parts:
            continue
        year = date_parts[0]
        month = date_parts[1] if len(date_parts) > 1 else 1
        day = date_parts[2] if len(date_parts) > 2 else 1
        try:
            return datetime(year, month, day, tzinfo=timezone.utc)
        except ValueError:
            continue
    return None

def normalize_key(paper: dict) -> Tuple[str, str]:
    title = re.sub(r"\s+", " ", (paper.get("title") or "").strip().lower())
    identifier = (paper.get("doi") or paper.get("link") or "").strip().lower()
    return title, identifier

def fetch_crossref_papers(year_start: int, year_end: int, needed: int, seen_keys: set,
                          keywords: Iterable[str] = None) -> List[dict]:
    """Backfill using CrossRef with citation counts."""
    if needed <= 0:
        return []
    fetched: List[dict] = []
    search_terms = list(keywords or ALL_KEYWORDS)
    for idx, phrase in enumerate(search_terms):
        if idx >= CROSSREF_MAX_KEYWORDS:
            break
        params = {
            "query": phrase,
            "rows": CROSSREF_ROWS_PER_KEYWORD,
            "filter": f"from-pub-date:{year_start}-01-01,until-pub-date:{year_end}-12-31",
            "select": "title,abstract,DOI,URL,author,issued,created,"
                      "published-print,published-online,deposited,subtitle,"
                      "container-title,is-referenced-by-count",
            "sort": "published",
            "order": "desc"
        }
        try:
            resp = request_with_backoff(CROSSREF_ENDPOINT, params=params, base_delay=1.0)
        except requests.RequestException as err:
            logger.warning("CrossRef request failed for '%s': %s", phrase, err)
            continue
        if not resp:
            continue
        items = resp.json().get("message", {}).get("items", [])
        candidates = []
        for item in items:
            title_list = item.get("title") or []
            title = title_list[0] if title_list else None
            if not title:
                continue
            published = extract_crossref_date(item)
            if not published:
                continue
            if not (year_start <= published.year <= year_end):
                continue
            doi = item.get("DOI")
            link = item.get("URL") or (f"https://doi.org/{doi}" if doi else "")
            key = normalize_key({"title": title, "doi": doi, "link": link})
            if key in seen_keys:
                continue
            abstract = item.get("abstract") or ""
            abstract = re.sub(r"<[^>]+>", "", abstract)
            if not abstract:
                subtitles = item.get("subtitle") or []
                abstract = " ".join(subtitles).strip()
            if not abstract:
                journal = ""
                container_titles = item.get("container-title") or []
                if container_titles:
                    journal = container_titles[0]
                abstract = f"Published in {journal} ({published.year})." if journal else f"Published in {published.year}."
            authors_data = item.get("author") or []
            authors = ", ".join(
                " ".join(filter(None, [a.get("given"), a.get("family")])).strip()
                for a in authors_data if a
            )
            citations = item.get("is-referenced-by-count", 0) or 0
            paper = {
                "title": title,
                "summary": abstract,
                "link": link or f"https://scholar.google.com/scholar?q={quote_plus(title)}",
                "published": published,
                "citations": citations,
                "authors": authors,
                "doi": f"https://doi.org/{doi}" if doi else link,
                "source": "crossref"
            }
            score = paper.get("citations", 0)
            candidates.append((score, paper, key))
        for _, paper, key in sorted(candidates, key=lambda x: x[0], reverse=True):
            if key in seen_keys:
                continue
            fetched.append(paper)
            seen_keys.add(key)
            if len(fetched) >= needed:
                break
        if len(fetched) >= needed:
            break
        time.sleep(1)
    return fetched

def fetch_scholar_papers(year_start: int, year_end: int, needed: int, seen_keys: set,
                         keywords: Iterable[str] = None) -> List[dict]:
    """Backfill using Google Scholar (if available) for additional coverage."""
    if needed <= 0 or scholarly is None:
        if scholarly is None:
            logger.warning("scholarly package not installed; skipping Google Scholar backfill.")
        return []
    fetched = []
    failure_count = 0
    search_terms = list(keywords or ALL_KEYWORDS)
    for idx, phrase in enumerate(search_terms):
        if idx >= SCHOLAR_MAX_KEYWORDS:
            break
        query = f'"{phrase}" after:{year_start - 1} before:{year_end + 1}'
        try:
            search = scholarly.search_pubs(query)
        except Exception as err:
            logger.warning("Scholar search failed for '%s': %s", phrase, err)
            failure_count += 1
            if failure_count >= SCHOLAR_MAX_FAILURES:
                break
            continue
        for pub in islice(search, SCHOLAR_RESULTS_PER_KEYWORD):
            try:
                detailed = scholarly.fill(pub)
            except Exception as err:
                logger.debug("Scholar fill error: %s", err)
                continue
            bib = detailed.get("bib", {})
            title = bib.get("title")
            if not title:
                continue
            try:
                year = int(bib.get("pub_year"))
            except (TypeError, ValueError):
                continue
            if not (year_start <= year <= year_end):
                continue
            link = (
                detailed.get("pub_url")
                or detailed.get("eprint_url")
                or bib.get("url")
                or f"https://scholar.google.com/scholar?q={quote_plus(title)}"
            )
            doi = bib.get("doi") or link
            key = normalize_key({"title": title, "doi": doi, "link": link})
            if key in seen_keys:
                continue
            authors_raw = bib.get("author", "")
            if isinstance(authors_raw, list):
                authors = ", ".join(authors_raw)
            else:
                authors = ", ".join(
                    a.strip() for a in authors_raw.split(" and ") if a.strip()
                ) if authors_raw else ""
            abstract = bib.get("abstract") or ""
            paper = {
                "title": title,
                "summary": abstract,
                "link": link,
                "published": datetime(year, 1, 1, tzinfo=timezone.utc),
                "citations": detailed.get("num_citations", 0) or 0,
                "authors": authors,
                "doi": doi if doi.startswith("http") else f"https://doi.org/{doi}" if doi else link,
                "source": "scholar"
            }
            fetched.append(paper)
            seen_keys.add(key)
            if len(fetched) >= needed:
                break
        if len(fetched) >= needed:
            break
        time.sleep(SCHOLAR_RATE_LIMIT_SECONDS)
    return fetched

def fetch_crossref_citation(doi: str) -> int:
    """Retrieve citation count for a DOI via CrossRef with caching."""
    if not doi:
        return 0
    doi = sanitize_doi(doi)
    if not doi:
        return 0
    if doi in CITATION_CACHE:
        return CITATION_CACHE[doi]
    url = f"https://api.crossref.org/works/{quote_plus(doi)}"
    try:
        resp = request_with_backoff(url, base_delay=1.0)
    except requests.RequestException:
        CITATION_CACHE[doi] = 0
        return 0
    if not resp:
        return 0
    count = resp.json().get("message", {}).get("is-referenced-by-count", 0) or 0
    CITATION_CACHE[doi] = count
    return count


def enrich_with_citations(paper: dict) -> None:
    """Ensure every paper has a citation count."""
    if paper.get("citations") is not None:
        return
    doi = paper.get("doi")
    if doi and isinstance(doi, str):
        citations = fetch_crossref_citation(doi)
        paper["citations"] = citations
    else:
        paper["citations"] = 0


def ensure_scores(papers: Iterable[dict], now: datetime, learned_weights: Dict[str, float]) -> None:
    for paper in papers:
        if not paper:
            continue
        if paper.get("score") is None:
            paper["score"] = enhanced_score(paper, now, learned_weights)

def backfill_time_window(
    current: List[dict],
    start_year: int,
    end_year: int,
    target: int,
    seen_keys: set,
    learned_weights: Dict[str, float],
    now: datetime,
) -> Tuple[List[dict], List[dict], List[dict]]:
    current = sorted(current, key=lambda x: x["score"], reverse=True)[:target]
    for paper in current:
        ensure_summary_text(paper)
    needed = target - len(current)
    crossref_extras: List[dict] = []
    scholar_extras: List[dict] = []
    if needed > 0:
        crossref_extras = fetch_crossref_papers(start_year, end_year, needed, seen_keys)
        for paper in crossref_extras:
            ensure_summary_text(paper)
            paper["score"] = enhanced_score(paper, now, learned_weights)
            current.append(paper)
            seen_keys.add(normalize_key(paper))
        needed = target - len(current)
    if needed > 0:
        scholar_extras = fetch_scholar_papers(start_year, end_year, needed, seen_keys)
        for paper in scholar_extras:
            ensure_summary_text(paper)
            paper["score"] = enhanced_score(paper, now, learned_weights)
            current.append(paper)
            seen_keys.add(normalize_key(paper))
    current = sorted(current, key=lambda x: x["score"], reverse=True)[:target]
    return current, crossref_extras, scholar_extras

def load_history_papers(exclude_links: set) -> Tuple[List[dict], set]:
    if not os.path.exists(LOG_FILE):
        return [], set()
    history: List[dict] = []
    sent_keys: set = set()
    with open(LOG_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            link = row.get("link") or ""
            if link in exclude_links:
                continue
            title = row.get("title", "")
            summary = row.get("summary", "")
            authors = row.get("authors", "")
            try:
                published = datetime.strptime(row.get("published", ""), "%Y-%m-%d").replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            try:
                score = float(row.get("relevance", 0))
            except (TypeError, ValueError):
                score = 0.0
            paper = {
             "title": title,
             "summary": summary,
             "link": link,
                "published": published,
                "citations": 0,
             "authors": authors,
                "doi": f"https://scholar.google.com/scholar?q={quote_plus(title)}",
                "source": "history",
                "score": score,
            }
            ensure_summary_text(paper)
            sent_keys.add(normalize_key(paper))
            history.append(paper)
    return history, sent_keys

def detect_trending_topics(papers: List[dict], now: datetime) -> List[str]:
    """Identify top bigram/trigram trends from the last 90 days."""
    cutoff = now - timedelta(days=90)
    texts = [
        f"{p['title']} {p.get('summary', '')}"
        for p in papers
        if p["published"] >= cutoff
    ]
    if not texts:
        return []
    vectorizer = CountVectorizer(ngram_range=(2, 3), stop_words="english", min_df=3)
    try:
        X = vectorizer.fit_transform(texts)
    except ValueError:
        return []
    counts = X.sum(axis=0).A1
    vocab = vectorizer.get_feature_names_out()
    phrases_with_counts = sorted(zip(vocab, counts), key=lambda x: x[1], reverse=True)
    keyword_set = set(k.lower() for k in ALL_KEYWORDS)
    trending = []
    for phrase, count in phrases_with_counts:
        if phrase.lower() in keyword_set:
            continue
        if not re.search(r"[a-z]", phrase):
            continue
        if len(trending) >= 5:
            break
        trending.append(f"{phrase} ({int(count)} hits)")
    return trending

def cluster_weekly_sections(papers: List[dict]) -> List[Tuple[str, List[dict]]]:
    """Group papers by keyword cluster for the weekly mode."""
    clusters: Dict[str, List[dict]] = defaultdict(list)
    for paper in papers:
        text = f"{paper['title']} {paper.get('summary', '')}".lower()
        best_group = "general"
        best_hits = 0
        for group, phrases in KEYWORD_GROUPS.items():
            hits = sum(phrase.lower() in text for phrase in phrases)
            if hits > best_hits:
                best_group = group
                best_hits = hits
        clusters[best_group].append(paper)
    sections = []
    for group, items in clusters.items():
        sections.append((group.replace("_", " ").title(), sorted(items, key=lambda x: x["score"], reverse=True)[:10]))
    return sorted(sections, key=lambda s: len(s[1]), reverse=True)

def format_score_bar(score: float) -> str:
    color = "#2ecc71" if score > 5 else "#3498db"
    width = min(100, max(10, int(score * 12)))
    return f"<div class='relevance-bar' style='background:{color};width:{width}%'>Score {score:.2f}</div>"

def generate_discovery_links(paper: dict) -> str:
    title = paper.get("title", "")
    doi = paper.get("doi") or ""
    doi_link = doi if doi.startswith("http") else f"https://doi.org/{doi}" if doi else ""
    scholar = f"https://scholar.google.com/scholar?q={quote_plus(title)}"
    semantic = f"https://www.semanticscholar.org/search?q={quote_plus(title)}"
    connected = f"https://www.connectedpapers.com/search?q={quote_plus(title)}"
    links = []
    if doi_link:
        links.append(f"<a href='{doi_link}'>DOI</a>")
    links.append(f"<a href='{paper.get('link', doi_link) or doi_link}'>Publisher</a>")
    links.append(f"<a href='{scholar}'>Google Scholar</a>")
    links.append(f"<a href='{semantic}'>Semantic Scholar</a>")
    links.append(f"<a href='{connected}'>Connected Papers</a>")
    return " | ".join(links)

def build_email_sections_html(sections: List[Tuple[str, List[dict]]],
                              weight_stats: Dict[str, float], mode: str) -> str:
    """Render the HTML email body with enhanced styling and metadata."""
    css = """
    <style>
      body { font-family: Arial, sans-serif; color:#1c2833; }
      h1 { color:#1a5276; }
      .section-title { margin-top:24px; border-bottom:2px solid #1a5276; padding-bottom:6px; }
      .paper-card { border:1px solid #d5d8dc; padding:12px 16px; margin:12px 0; border-radius:8px; background:#fdfefe; }
      .relevance-bar { height:14px; border-radius:6px; color:#fff; font-size:11px; line-height:14px; text-align:center; margin-bottom:8px; }
      .meta { color:#566573; font-size:13px; margin-bottom:6px; }
      .links { font-size:13px; margin-top:6px; }
      .summary { margin-top:8px; }
      .source-badge { display:inline-block; padding:2px 6px; border-radius:4px; font-size:11px; color:#fff; margin-left:8px; }
      .source-feed { background:#1abc9c; }
      .source-crossref { background:#9b59b6; }
      .source-scholar { background:#e67e22; }
      .source-history { background:#95a5a6; }
      .source-adjacent { background:#34495e; }
      .footer { font-size:12px; color:#7b7d7d; margin-top:24px; }
    </style>
    """
    body = [css, "<h1>Daily Paper Digest</h1>"]
    if mode == "weekly":
        body[1] = "<h1>Weekly Paper Digest</h1>"
    for label, items in sections:
        body.append(f"<h2 class='section-title'>{label}</h2>")
        if not items:
            body.append("<p>No papers found for this section today.</p>")
            continue
        for paper in items:
            score_bar = format_score_bar(paper["score"])
            meta = f"{paper['published'].strftime('%Y-%m-%d')} · {paper.get('authors','Unknown')}"
            badge_class = f"source-{paper.get('source','adjacent')}"
            badge_label = paper.get("source", "Unknown").title()
            summary = short_summary(paper.get("summary", ""))
            if paper["score"] > 5:
                summary = abstract_preview(paper.get("summary", ""))
            links = generate_discovery_links(paper)
            body.append(
                "<div class='paper-card'>"
                f"<div style='display:flex;align-items:center;justify-content:space-between;'>"
                f"<div style='font-size:16px;font-weight:bold;'>{paper['title']}</div>"
                f"<span class='source-badge {badge_class}'>{badge_label}</span>"
                "</div>"
                f"{score_bar}"
                f"<div class='meta'>{meta} · Citations: {paper.get('citations',0)}</div>"
                f"<div class='links'>{links}</div>"
                f"<div class='summary'>{summary}</div>"
                "</div>"
            )
    if weight_stats:
        top_weights = sorted(weight_stats.items(), key=lambda kv: kv[1], reverse=True)[:5]
        stats_str = ", ".join(f"{phrase}: {val:.2%}" for phrase, val in top_weights)
        body.append(f"<div class='footer'>Personalization signals from recent clicks: {stats_str}</div>")
    return "\n".join(body)

def log_papers(sections: List[Tuple[str, List[dict]]]) -> None:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    header = ["date", "section", "title", "published", "link", "relevance", "summary", "authors"]
    newfile = not os.path.exists(LOG_FILE)
    existing_keys = set()
    rows: List[Tuple[str, str, dict]] = []
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_keys.add((row.get("section", ""), normalize_key({"title": row.get("title",""), "link": row.get("link",""), "doi": ""})))
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if newfile:
            writer.writerow(header)
        for label, items in sections:
            for it in items:
                key = (label, normalize_key(it))
                if key in existing_keys:
                    continue
                existing_keys.add(key)
                rows.append((label, it))
        for label, it in rows:
            writer.writerow([
                today,
                label,
                it.get("title"),
                it.get("published", datetime.now(timezone.utc)).strftime("%Y-%m-%d"),
                it.get("link"),
                it.get("score"),
                short_summary(it.get("summary", "")),
                it.get("authors")
            ])
    logger.info("Logged %d new papers.", len(rows))

def build_rss_feed(sections: List[Tuple[str, List[dict]]]) -> None:
    """Generate RSS 2.0 feed from digest sections."""
    from xml.etree.ElementTree import Element, SubElement, tostring
    from xml.dom import minidom
    
    now = datetime.now(timezone.utc)
    
    # Create RSS root
    rss = Element('rss', version='2.0')
    rss.set('xmlns:atom', 'http://www.w3.org/2005/Atom')
    
    channel = SubElement(rss, 'channel')
    
    # Channel metadata
    SubElement(channel, 'title').text = 'Paper Digest - Daily Research Papers'
    SubElement(channel, 'link').text = 'https://github.com/ddrizzil/paper-digest'
    SubElement(channel, 'description').text = 'Curated daily digest of research papers in cultural heritage, imaging, RF systems, signal processing, and related fields'
    SubElement(channel, 'language').text = 'en-us'
    SubElement(channel, 'lastBuildDate').text = now.strftime('%a, %d %b %Y %H:%M:%S %z')
    SubElement(channel, 'pubDate').text = now.strftime('%a, %d %b %Y %H:%M:%S %z')
    SubElement(channel, 'generator').text = 'Paper Digest Service'
    
    # Atom self link
    atom_link = SubElement(channel, 'atom:link')
    atom_link.set('href', 'https://raw.githubusercontent.com/ddrizzil/paper-digest/main/feed.xml')
    atom_link.set('rel', 'self')
    atom_link.set('type', 'application/rss+xml')
    
    # Collect all papers from all sections
    all_papers = []
    for section_label, papers in sections:
        for paper in papers:
            # Add section info to paper for context
            paper_with_section = paper.copy()
            paper_with_section['section'] = section_label
            all_papers.append(paper_with_section)
    
    # Sort by score (relevance) descending, then by publication date
    all_papers.sort(key=lambda p: (
        -float(p.get('score', 0) or 0),
        -(p.get('published', now).timestamp() if isinstance(p.get('published'), datetime) else now.timestamp())
    ))
    
    # Limit to most recent 50 items for RSS feed
    all_papers = all_papers[:50]
    
    # Add items
    for paper in all_papers:
        item = SubElement(channel, 'item')
        
        title = paper.get('title', 'Untitled')
        SubElement(item, 'title').text = title
        
        link = paper.get('link', '')
        if link:
            SubElement(item, 'link').text = link
        
        # Description with summary and metadata
        description_parts = []
        summary = paper.get('summary', '').strip()
        if summary:
            description_parts.append(summary)
        
        authors = paper.get('authors', '')
        if authors:
            description_parts.append(f"Authors: {authors}")
        
        section = paper.get('section', '')
        if section:
            description_parts.append(f"Category: {section}")
        
        score = paper.get('score', 0)
        if score:
            description_parts.append(f"Relevance Score: {score:.2f}")
        
        citations = paper.get('citations', 0)
        if citations:
            description_parts.append(f"Citations: {citations}")
        
        description = ' | '.join(description_parts) if description_parts else 'No description available'
        # Escape XML special characters
        import html
        description = html.escape(description)
        desc_elem = SubElement(item, 'description')
        desc_elem.text = description
        
        # Publication date
        published = paper.get('published')
        if published:
            if isinstance(published, datetime):
                pub_date = published
            else:
                try:
                    pub_date = parsedate_to_datetime(str(published))
                except:
                    pub_date = now
        else:
            pub_date = now
        
        if pub_date.tzinfo is None:
            pub_date = pub_date.replace(tzinfo=timezone.utc)
        
        SubElement(item, 'pubDate').text = pub_date.strftime('%a, %d %b %Y %H:%M:%S %z')
        
        # GUID (use link or title+doi)
        guid = SubElement(item, 'guid')
        doi = paper.get('doi', '')
        if doi:
            guid.text = doi
        elif link:
            guid.text = link
        else:
            guid.text = f"paper-digest-{hash(title)}"
        guid.set('isPermaLink', 'false' if doi else ('true' if link else 'false'))
        
        # Author
        if authors:
            # RSS 2.0 uses email format, but we'll use name
            author_text = authors.split(',')[0].strip() if ',' in authors else authors.strip()
            SubElement(item, 'author').text = author_text
        
        # Category (section)
        if section:
            SubElement(item, 'category').text = section
    
    # Format XML nicely
    xml_str = tostring(rss, encoding='unicode')
    dom = minidom.parseString(xml_str)
    pretty_xml = dom.toprettyxml(indent='  ', encoding=None)
    
    # Remove the XML declaration line that minidom adds (we'll add our own)
    lines = pretty_xml.split('\n')
    if lines and lines[0].startswith('<?xml'):
        lines = lines[1:]
    pretty_xml = '\n'.join(lines).strip()
    
    # Write RSS feed
    feed_path = Path('feed.xml')
    with open(feed_path, 'w', encoding='utf-8') as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write(pretty_xml)
    
    logger.info("RSS feed generated with %d items: %s", len(all_papers), feed_path)

def build_html_archive() -> None:
    import pandas as pd  # Lazy import to avoid dependency if not installed.
    if not os.path.exists(LOG_FILE):
        return
    df = pd.read_csv(LOG_FILE)
    df = df.sort_values(["date", "section"], ascending=[False, True])
    before = len(df)
    df = df.drop_duplicates(subset=["date", "section", "title", "link"], keep="first")
    after = len(df)
    if after != before:
        df.sort_values(["date", "section"], ascending=[False, True]).to_csv(LOG_FILE, index=False)
        logger.info("Deduplicated %d duplicate log rows.", before - after)
    html = [
        "<html><head><meta charset='utf-8'><title>Paper Archive</title>",
          "<style>body{font-family:sans-serif;padding:20px;}table{width:100%;border-collapse:collapse;}th,td{border:1px solid #ccc;padding:6px;}</style>",
        "</head><body><h1>📚 Paper Digest Archive</h1>"
    ]
    for date, g in df.groupby("date"):
        html.append(f"<h2>{date}</h2>")
        for section, sg in g.groupby("section"):
            html.append("<h3>{}</h3><table><tr><th>Title</th><th>Date</th><th>Rel.</th><th>Summary</th><th>Authors</th></tr>".format(section))
            for _, r in sg.iterrows():
                author_link = ""
                if pd.notna(r.authors) and r.authors:
                    first_author = r.authors.split(",")[0]
                    author_link = f"https://scholar.google.com/scholar?q={quote_plus(first_author)}"
                dive_link = f"https://scholar.google.com/scholar?q={quote_plus(r.title)}"
                html.append(
                    "<tr>"
                    f"<td><a href='{r.link}'>{r.title}</a><br>"
                            f"<a href='{dive_link}'>Dive deeper</a>"
                            + (f" | <a href='{author_link}'>Author page</a>" if author_link else "")
                    + "</td>"
                    f"<td>{r.published}</td><td>{r.relevance}</td><td>{r.summary}</td><td>{r.authors}</td>"
                    "</tr>"
                )
            html.append("</table>")
    html.append("</body></html>")
    with open("logs/archive.html", "w", encoding="utf-8") as f:
        f.write("\n".join(html))
    logger.info("Archive HTML updated.")

def email_digest(sections: List[Tuple[str, List[dict]]],
                 weight_stats: Dict[str, float], mode: str) -> None:
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    smtp_host = os.getenv("SMTP_HOST")
    smtp_port_str = os.getenv("SMTP_PORT", "").strip()
    if not smtp_port_str:
        smtp_port = 587
    else:
        try:
            smtp_port = int(smtp_port_str)
        except (ValueError, TypeError):
            smtp_port = 587
    smtp_user = os.getenv("SMTP_USER")
    smtp_pass = os.getenv("SMTP_PASS")
    recipients = load_recipients()

    # Diagnostic logging
    logger.info("Email digest check - SMTP_HOST: %s, SMTP_PORT: %s, SMTP_USER: %s, Recipients: %d", 
                "SET" if smtp_host else "NOT SET",
                smtp_port,
                "SET" if smtp_user else "NOT SET",
                len(recipients))
    
    if not recipients:
        logger.error("No email recipients found! Check EMAIL_RECIPIENTS secret or config/emails.txt file.")
        return

    msg = MIMEMultipart("alternative")
    subject_prefix = "Weekly" if mode == "weekly" else "Daily"
    msg["Subject"] = f"📚 {subject_prefix} Paper Digest – {datetime.now(timezone.utc).strftime('%Y-%m-%d')}"
    msg["From"] = SENDER
    msg["To"] = ", ".join(recipients)

    html_body = build_email_sections_html(sections, weight_stats, mode)
    msg.attach(MIMEText(html_body, "html"))

    if not smtp_host:
        logger.warning("SMTP_HOST is not set; skipping email send.")
        return

    try:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            logger.info("Connecting to SMTP server %s:%d", smtp_host, smtp_port)
            server.starttls()
            if not smtp_user or not smtp_pass:
                logger.warning("SMTP credentials missing; skipping email send.")
                return
            logger.info("Logging in to SMTP server as %s", smtp_user)
            server.login(smtp_user, smtp_pass)
            logger.info("Sending email to: %s", ", ".join(recipients))
            server.sendmail(SENDER, recipients, msg.as_string())
        logger.info("Email successfully sent to %s", ", ".join(recipients))
    except Exception as e:
        logger.error("Failed to send email: %s", str(e), exc_info=True)
        raise

def has_priority_topic(paper: dict) -> bool:
    text = f"{paper.get('title', '')} {paper.get('summary', '')}".lower()
    return any(keyword in text for keyword in PRIORITY_ML_KEYWORDS)


def classify_interest_group(paper: dict) -> str:
    text = f"{paper.get('title', '')} {paper.get('summary', '')}".lower()
    for group in ML_RF_GROUPS:
        for phrase in KEYWORD_GROUPS.get(group, []):
            if phrase.lower() in text:
                return "ml_rf"
    for group in HERITAGE_OPTICS_GROUPS:
        for phrase in KEYWORD_GROUPS.get(group, []):
            if phrase.lower() in text:
                return "heritage_optics"
    return "other"

# ---------------- MAIN PROCESS ----------------
def run_once() -> None:
    now = datetime.now(timezone.utc)
    clicked_titles = load_click_history()
    learned_weights, weight_stats = update_keyword_weights(clicked_titles)

    papers: List[dict] = []
    seen_links = set()

    for url in FEEDS:
        for entry in parse_feed(url):
            title, summary = entry.get("title", ""), entry.get("summary", "")
            link = entry.get("link", "")
            if link and link in seen_links:
                continue
            raw_authors = entry.get("authors", [])
            if isinstance(raw_authors, list):
                authors = ", ".join([a.get("name", "") for a in raw_authors])
            else:
                authors = str(raw_authors)
            pub_date = extract_pub_date(entry)
            if not pub_date:
                continue
            doi = entry.get("doi") if isinstance(entry, dict) and entry.get("doi") else f"https://scholar.google.com/scholar?q={title.replace(' ', '+')}"
            paper = {
                "title": title,
                "summary": summary,
                "link": link,
                "published": pub_date,
                "citations": None,  # to be filled
                "authors": authors,
                "doi": doi,
                "source": "feed"
            }
            if link:
                seen_links.add(link)
            enrich_with_citations(paper)
            ensure_summary_text(paper)
            paper["score"] = enhanced_score(paper, now, learned_weights)
            papers.append(paper)

    history_papers, sent_history_keys = load_history_papers(seen_links)
    all_papers = papers + history_papers
    seen_keys = {normalize_key(p) for p in all_papers}

    sent_keys_all = set(sent_history_keys)
    used_keys_run: set = set()

    def register_selected(selected: List[dict]) -> None:
        for entry in selected:
            key = normalize_key(entry)
            used_keys_run.add(key)
            sent_keys_all.add(key)

    # Evaluate scores for history entries using current weights.
    for paper in history_papers:
        if paper.get("score") in (None, 0):
            paper["score"] = enhanced_score(paper, now, learned_weights)

    all_papers = [p for p in all_papers if p.get("published")]
    ensure_scores(all_papers, now, learned_weights)

    one_year_cutoff = now - timedelta(days=365)
    ten_year_cutoff = now - timedelta(days=10 * 365)

    recent_pool = [p for p in all_papers if p["published"] >= one_year_cutoff]
    recent_pool = sorted(recent_pool, key=lambda x: x["score"], reverse=True)

    ml_candidates = [p for p in recent_pool if classify_interest_group(p) == "ml_rf"]
    heritage_candidates = [p for p in recent_pool if classify_interest_group(p) == "heritage_optics"]
    other_candidates = [p for p in recent_pool if classify_interest_group(p) == "other"]

    priority_ml = [p for p in ml_candidates if has_priority_topic(p)]
    nonpriority_ml = [p for p in ml_candidates if not has_priority_topic(p)]

    recent_keys: set = set()
    ml_recent: List[dict] = []
    heritage_recent: List[dict] = []
    half_target = RECENT_TARGET // 2 or 1

    def fill_recent_pool(source: List[dict], dest: List[dict], quota: int) -> None:
        for paper in source:
            if len(dest) >= quota:
                break
            key = normalize_key(paper)
            if key in sent_keys_all or key in used_keys_run or key in recent_keys:
                continue
            dest.append(paper)
            recent_keys.add(key)

    fill_recent_pool(priority_ml, ml_recent, half_target)
    fill_recent_pool(nonpriority_ml, ml_recent, half_target)
    fill_recent_pool(heritage_candidates, heritage_recent, half_target)

    recent = ml_recent + heritage_recent
    for source in (priority_ml, nonpriority_ml, heritage_candidates, other_candidates):
        if len(recent) >= RECENT_TARGET:
            break
        for paper in source:
            if len(recent) >= RECENT_TARGET:
                break
            key = normalize_key(paper)
            if key in sent_keys_all or key in used_keys_run or key in recent_keys:
                continue
            recent.append(paper)
            recent_keys.add(key)

    register_selected(recent)

    decade_candidates = [
        p for p in all_papers
        if ten_year_cutoff <= p["published"] < one_year_cutoff
    ]
    last_decade, decade_crossref, decade_scholar = backfill_time_window(
        decade_candidates,
        start_year=now.year - 10,
        end_year=now.year - 1,
        target=DECADE_TARGET,
        seen_keys=seen_keys,
        learned_weights=learned_weights,
        now=now,
    )
    all_papers.extend(decade_crossref + decade_scholar)
    last_decade = [p for p in last_decade if p["score"] >= DECADE_MIN_SCORE]
    decade_filtered: List[dict] = []
    decade_keys: set = set()
    for paper in sorted(last_decade, key=lambda x: x["score"], reverse=True):
        key = normalize_key(paper)
        if key in sent_keys_all or key in used_keys_run or key in decade_keys:
            continue
        decade_filtered.append(paper)
        decade_keys.add(key)
        if len(decade_filtered) >= DECADE_TARGET:
            break
    last_decade = decade_filtered
    register_selected(last_decade)

    adjacent: List[dict] = []
    adjacent_seen: set = set()

    def add_adjacent_candidate(paper: dict) -> None:
        key = normalize_key(paper)
        if key in sent_keys_all or key in used_keys_run or key in adjacent_seen:
            return
        adjacent.append(paper)
        adjacent_seen.add(key)

    trigger_keywords = [
        "nonlinear", "radar", "climate", "environment", "biodiversity",
        "sustainability", "commercial radar", "art conservation", "optical"
    ]
    candidate_pool = sorted(all_papers, key=lambda x: (x["published"], x["score"]), reverse=True)
    for candidate in candidate_pool:
        text = (candidate["title"] + " " + candidate.get("summary", "")).lower()
        if any(word in text for word in trigger_keywords):
            add_adjacent_candidate(candidate)
        if len(adjacent) >= ADJACENT_TARGET:
            break

    adjacent_crossref: List[dict] = []
    adjacent_scholar: List[dict] = []
    if len(adjacent) < ADJACENT_TARGET:
        needed_adj = ADJACENT_TARGET - len(adjacent)
        adjacent_crossref = fetch_crossref_papers(
            now.year - 7,
            now.year,
            needed_adj,
            seen_keys,
            keywords=ADJACENT_KEYWORDS,
        )
        for paper in adjacent_crossref:
            ensure_summary_text(paper)
            paper["score"] = enhanced_score(paper, now, learned_weights)
            add_adjacent_candidate(paper)
            all_papers.append(paper)
    if len(adjacent) < ADJACENT_TARGET and scholarly is not None:
        needed_adj = ADJACENT_TARGET - len(adjacent)
        adjacent_scholar = fetch_scholar_papers(
            now.year - 7,
            now.year,
            needed_adj,
            seen_keys,
            keywords=ADJACENT_KEYWORDS,
        )
        for paper in adjacent_scholar:
            ensure_summary_text(paper)
            paper["score"] = enhanced_score(paper, now, learned_weights)
            add_adjacent_candidate(paper)
            all_papers.append(paper)

    adjacent = sorted(adjacent, key=lambda x: x["score"], reverse=True)[:ADJACENT_TARGET]

    register_selected(adjacent)

    all_papers.extend(adjacent_crossref + adjacent_scholar)

    sections = [
        ("Recent (≤1 year)", recent),
        ("Last 10 years", last_decade),
        ("Adjacent opportunities", adjacent)
    ]

    dedup_sections = []
    seen_section_keys = set()
    for label, items in sections:
        unique_items = []
        for paper in items:
            key = (label, normalize_key(paper))
            if key in seen_section_keys:
                continue
            seen_section_keys.add(key)
            unique_items.append(paper)
        dedup_sections.append((label, unique_items))

    if DIGEST_MODE == "weekly":
        dedup_sections = cluster_weekly_sections(all_papers)

    logger.info(
        "Paper pools – rss:%d history:%d decade_crossref:%d decade_scholar:%d adjacent_crossref:%d adjacent_scholar:%d recent:%d decade:%d adjacent:%d",
        len(papers),
        len(history_papers),
        len(decade_crossref),
        len(decade_scholar),
        len(adjacent_crossref),
        len(adjacent_scholar),
        len(recent),
        len(last_decade),
        len(adjacent)
    )

    log_papers(dedup_sections)
    build_html_archive()
    build_rss_feed(dedup_sections)
    email_digest(dedup_sections, weight_stats, DIGEST_MODE)

if __name__ == "__main__":
    run_once()