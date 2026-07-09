import os
from dotenv import load_dotenv

load_dotenv()

# ── Model ────────────────────────────────────────────────────────────────────
MODEL_NAME = "qwen2.5:3b"

# ── API Keys ─────────────────────────────────────────────────────────────────
OPENWEATHERMAP_API_KEY = os.environ.get("OPENWEATHERMAP_API_KEY")

# ── Search ───────────────────────────────────────────────────────────────────
NUM_SEARCH_RESULTS = 3          # Number of search results to return
SEARXNG_URL = "http://localhost:8080" # URL of the SearXNG instance

# ── Page Scraping ────────────────────────────────────────────────────────────
MAX_PAGE_CONTENT_LENGTH = 5000  # Max characters to extract from a single page
PAGE_REQUEST_TIMEOUT = 10       # Seconds before a page fetch times out

# ── Agent Loop ───────────────────────────────────────────────────────────────
MAX_TOOL_ITERATIONS = 10        # Safety cap: max tool-call round-trips per turn

# ── Search Cache (ChromaDB) ─────────────────────────────────────────────────
CACHE_DIR = "~/.chatbot_cache/chroma_db"   # Persistent ChromaDB storage path
CACHE_TTL_SECONDS = 86_400                  # Cache validity: 24 hours
CACHE_SIMILARITY_THRESHOLD = 0.35           # Max L2 distance for a cache hit
                                            #   Lower = stricter, higher = looser

# ── Biziday News Index (ChromaDB) ───────────────────────────────────────
BIZIDAY_COLLECTION_NAME = "biziday_news"  # Separate collection for news articles
BIZIDAY_SEARCH_RESULTS = 5               # Default number of semantic search results
BIZIDAY_RELEVANCE_THRESHOLD = 1.0        # Max L2 distance to consider a match "relevant"
                                         #   (tighter = fewer but more relevant results)

# ── Translation (SeamlessM4T) ───────────────────────────────────────────────
SEAMLESS_MODEL_LARGE = "facebook/seamless-m4t-v2-large"     # 2.3B params
SEAMLESS_MODEL_FALLBACK = "facebook/seamless-m4t-medium"    # 1.2B params