"""
ProTV → ChromaDB ingestion script.
Called by UiPath after scraping. Reads scraped_articles.csv,
translates RO → EN, chunks + embeds text, upserts into the
shared 'news' ChromaDB collection.
"""

import csv
import hashlib
import logging
import os
import sys

# Force UTF-8 encoding for the console to avoid crashes with emojis and diacritics
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

from datetime import datetime

import chromadb
from sentence_transformers import SentenceTransformer

# ── Configuration ────────────────────────────────────────────────────────
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
CSV_PATH      = os.path.join(SCRIPT_DIR, "scraped_articles.csv")
LOG_PATH      = os.path.join(SCRIPT_DIR, "ingest.log")

# Shared ChromaDB instance (same as your chatbot project)
CHROMA_PATH   = os.path.expanduser(r"~\.chatbot_cache\chroma_db")
COLLECTION    = "news"           # merged collection (Biziday + ProTV)

CHUNK_SIZE    = 800              # characters per chunk
CHUNK_OVERLAP = 100
BATCH_SIZE    = 64               # embeddings per batch
MODEL_NAME    = "all-MiniLM-L6-v2"

# Translation — set to True to translate RO → EN before embedding
TRANSLATE     = True
# Path to the chatbot project (for importing the translator module)
CHATBOT_DIR   = r"C:\Users\David\Chatbot_Internet_Searches"

# ── Logging ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    text = text.strip()
    if not text:
        return chunks
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks


def translate_text(text: str) -> str:
    """Translate Romanian text to English using the chatbot's translator."""
    if not TRANSLATE or not text.strip():
        return text
    try:
        # Import from the chatbot project
        if CHATBOT_DIR not in sys.path:
            sys.path.insert(0, CHATBOT_DIR)
        from translator import _chunk_text as chunk_for_translation, _translate
        
        chunks = chunk_for_translation(text)
        translated = [_translate(c, src_lang="ron", tgt_lang="eng") for c in chunks]
        return "\n\n".join(translated)
    except Exception as exc:
        log.warning("Translation failed, using original text: %s", exc)
        return text


def main():
    # ── Load CSV ─────────────────────────────────────────────────────
    if not os.path.exists(CSV_PATH):
        log.warning("No CSV found at %s — nothing to ingest.", CSV_PATH)
        return

    with open(CSV_PATH, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        log.info("CSV is empty — nothing to ingest.")
        return

    log.info("Loaded %d article(s) from CSV.", len(rows))

    # ── Translate + prepare chunks ───────────────────────────────────
    ids, documents, metadatas = [], [], []

    for row in rows:
        url       = row.get("URL", "").strip()
        title_ro  = row.get("Title", "").strip()
        category  = row.get("Category", "").strip()
        scraped   = row.get("ScrapedAt", datetime.now().isoformat())
        full_text = row.get("FullText", "").strip()

        if not full_text:
            log.warning("Skipping article with empty body: %s", url)
            continue

        # Translate
        log.info("Translating: %s", title_ro[:60])
        title_en = translate_text(title_ro)
        body_en  = translate_text(full_text)

        # The embedded document is the English version
        document_text = f"{title_en}\n\n{body_en}"

        for i, chunk in enumerate(chunk_text(document_text)):
            chunk_id = hashlib.sha256(f"{url}#{i}".encode()).hexdigest()
            ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append({
                "url": url,
                "title_ro": title_ro,
                "title_en": title_en,
                "category": category,
                "scraped_at": scraped,
                "chunk_index": i,
                "source": "stirileprotv.ro",
            })

    if not ids:
        log.info("No chunks to embed.")
        return

    log.info("Prepared %d chunk(s) from %d article(s).", len(ids), len(rows))

    # ── Embed ────────────────────────────────────────────────────────
    log.info("Loading embedding model '%s'...", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)

    log.info("Encoding %d chunks...", len(documents))
    embeddings = model.encode(
        documents, batch_size=BATCH_SIZE, show_progress_bar=True
    ).tolist()

    # ── Upsert into ChromaDB ─────────────────────────────────────────
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "l2"},
    )

    # ChromaDB batch limit — split into groups of 5000
    for start in range(0, len(ids), 5000):
        end = start + 5000
        collection.upsert(
            ids=ids[start:end],
            documents=documents[start:end],
            embeddings=embeddings[start:end],
            metadatas=metadatas[start:end],
        )

    log.info(
        "Done. Upserted %d chunks from %d articles. Collection total: %d.",
        len(ids), len(rows), collection.count(),
    )


if __name__ == "__main__":
    main()
