"""
Usage:
    python index_biziday.py                          # index default JSON
    python index_biziday.py --input path/to/file.json
    python index_biziday.py --search "Microsoft layoffs"
    python index_biziday.py --stats
    python index_biziday.py --clear
"""

import hashlib
import json
import os
import sys
import argparse

import chromadb

from config import (
    CACHE_DIR,
    BIZIDAY_COLLECTION_NAME,
    BIZIDAY_SEARCH_RESULTS,
)
from translator import translate_to_english, _chunk_text, _translate
from scrape_biziday import scrape_homepage, extract_article_text
from datetime import datetime

# Fix console encoding for Romanian characters (Windows only)
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Terminal colours
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
DIM    = "\033[2m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

DEFAULT_JSON_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "stiriBiziday.json"
)

# ChromaDB metadata values are limited to ~32KB.  We truncate the
# Romanian body to stay safely within that limit.
_MAX_METADATA_BODY_CHARS = 20_000


# ═══════════════════════════════════════════════════════════════════════════
#  ChromaDB Client
# ═══════════════════════════════════════════════════════════════════════════

def _get_collection() -> chromadb.Collection:
    """Return the persistent ChromaDB collection for Biziday articles."""
    persist_dir = os.path.expanduser(CACHE_DIR)
    client = chromadb.PersistentClient(path=persist_dir)
    return client.get_or_create_collection(
        name=BIZIDAY_COLLECTION_NAME,
        metadata={"hnsw:space": "l2"},
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Translation Helper
# ═══════════════════════════════════════════════════════════════════════════

def _translate_article(text: str) -> str:
    """Translate a Romanian article body to English using chunked translation.

    Long articles are split into manageable chunks, translated individually,
    and reassembled.  This mirrors the approach in translator.py's
    translate_to_romanian() but goes Romanian → English.
    """
    chunks = _chunk_text(text)
    translated_parts: list[str] = []

    for chunk in chunks:
        translated_parts.append(
            _translate(chunk, src_lang="ron", tgt_lang="eng")
        )

    return "\n\n".join(translated_parts)


# ═══════════════════════════════════════════════════════════════════════════
#  Indexing
# ═══════════════════════════════════════════════════════════════════════════

def _make_id(url: str) -> str:
    """Deterministic document ID from an article URL."""
    return hashlib.sha256(url.strip().encode()).hexdigest()[:16]


def index_articles(json_path: str = DEFAULT_JSON_PATH) -> int:
    """Load articles from JSON, translate to English, and index in ChromaDB.

    Returns the number of articles indexed.
    """
    print(f"\n{BOLD}📰  Biziday News Indexer{RESET}")
    print(f"{DIM}   JSON path: {json_path}{RESET}\n")

    # ── Load JSON ────────────────────────────────────────────────────────
    if not os.path.exists(json_path):
        print(f"{RED}✖  File not found: {json_path}{RESET}")
        return 0

    with open(json_path, "r", encoding="utf-8") as f:
        articles = json.load(f)

    if not articles:
        print(f"{YELLOW}⚠  No articles found in {json_path}{RESET}")
        return 0

    print(f"   Found {BOLD}{len(articles)}{RESET} articles to index\n")

    # ── Get ChromaDB collection ──────────────────────────────────────────
    collection = _get_collection()

    # ── Translate and index each article ─────────────────────────────────
    indexed = 0
    for i, article in enumerate(articles):
        title_ro = article.get("Titlu", "")
        body_ro = article.get("HTML_Sursa", "")
        url = article.get("URL", "")
        timestamp = article.get("Timestamp", "")

        if not url or not title_ro:
            print(f"  {YELLOW}[{i+1}] Skipping — missing URL or title{RESET}")
            continue

        title_preview = title_ro[:80]
        print(f"  [{i+1}/{len(articles)}] {title_preview}...")

        # ── Translate ────────────────────────────────────────────────────
        print(f"  {DIM}   🌍 Translating title...{RESET}")
        title_en = _translate(title_ro, src_lang="ron", tgt_lang="eng")

        print(f"  {DIM}   🌍 Translating article body...{RESET}")
        body_en = _translate_article(body_ro)

        # The full English text is what gets embedded
        document_text = f"{title_en}\n\n{body_en}"

        # Preview
        en_preview = title_en[:100]
        print(f"  {DIM}   → EN: {en_preview}...{RESET}")

        # ── Truncate Romanian body for metadata storage ──────────────────
        body_ro_stored = body_ro
        if len(body_ro_stored) > _MAX_METADATA_BODY_CHARS:
            body_ro_stored = body_ro_stored[:_MAX_METADATA_BODY_CHARS] + "…[truncated]"

        # ── Upsert into ChromaDB ─────────────────────────────────────────
        doc_id = _make_id(url)
        collection.upsert(
            ids=[doc_id],
            documents=[document_text],
            metadatas=[{
                "title_ro": title_ro,
                "title_en": title_en,
                "url": url,
                "timestamp": timestamp,
                "body_ro": body_ro_stored,
                "source": "biziday.ro",
            }],
        )
        indexed += 1

    print(f"\n{GREEN}✅  Indexed {indexed} articles into '{BIZIDAY_COLLECTION_NAME}' collection{RESET}")
    print(f"   Collection total: {collection.count()} documents\n")
    return indexed


def sync_latest_articles(count: int = 20, json_path: str = DEFAULT_JSON_PATH) -> int:
    """Scrape the homepage for new articles, check against ChromaDB, and index any missing ones.
    Optionally appends to the JSON file to keep it in sync.
    """
    print(f"\n{DIM}🔄  Checking for new Biziday articles...{RESET}")
    
    try:
        latest_articles = scrape_homepage(count)
    except Exception as exc:
        print(f"  {RED}✖  Failed to fetch homepage: {exc}{RESET}")
        return 0
        
    if not latest_articles:
        return 0

    collection = _get_collection()
    
    # Load existing JSON if we want to append
    existing_json = []
    if os.path.exists(json_path):
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                existing_json = json.load(f)
        except Exception:
            pass
            
    new_articles_count = 0
    
    for item in latest_articles:
        url = item["url"]
        headline_ro = item["headline"]
        doc_id = _make_id(url)
        
        # Check if it already exists in ChromaDB
        result = collection.get(ids=[doc_id])
        if result and result["ids"]:
            # Already exists, skip
            continue
            
        print(f"  {GREEN}✦  New article found:{RESET} {headline_ro[:80]}...")
        
        # Scrape full text
        print(f"  {DIM}   📄 Scraping full text...{RESET}")
        body_ro = extract_article_text(url)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Translate
        print(f"  {DIM}   🌍 Translating title...{RESET}")
        title_en = _translate(headline_ro, src_lang="ron", tgt_lang="eng")

        print(f"  {DIM}   🌍 Translating article body...{RESET}")
        body_en = _translate_article(body_ro)
        
        document_text = f"{title_en}\n\n{body_en}"
        
        body_ro_stored = body_ro
        if len(body_ro_stored) > _MAX_METADATA_BODY_CHARS:
            body_ro_stored = body_ro_stored[:_MAX_METADATA_BODY_CHARS] + "…[truncated]"
            
        # Upsert to ChromaDB
        collection.upsert(
            ids=[doc_id],
            documents=[document_text],
            metadatas=[{
                "title_ro": headline_ro,
                "title_en": title_en,
                "url": url,
                "timestamp": timestamp,
                "body_ro": body_ro_stored,
                "source": "biziday.ro",
            }],
        )
        
        # Append to JSON list
        existing_json.insert(0, {
            "Titlu": headline_ro,
            "HTML_Sursa": body_ro,
            "URL": url,
            "Timestamp": timestamp,
        })
        
        new_articles_count += 1

    if new_articles_count > 0:
        # Save updated JSON
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(existing_json, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            print(f"  {YELLOW}⚠  Could not update JSON backup: {exc}{RESET}")
            
        print(f"  {GREEN}✅  Added {new_articles_count} new articles to ChromaDB.{RESET}")
    else:
        print(f"  {DIM}✓  No new articles found. DB is up to date.{RESET}")
        
    return new_articles_count


# ═══════════════════════════════════════════════════════════════════════════
#  Search
# ═══════════════════════════════════════════════════════════════════════════

def search_biziday(query: str, n_results: int = BIZIDAY_SEARCH_RESULTS) -> list[dict]:
    """Search indexed Biziday articles by semantic similarity.

    Args:
        query: The search query (should be in English for best results).
        n_results: Maximum number of results to return.

    Returns:
        A list of dicts with keys: title_en, title_ro, url, timestamp,
        body_en (the document text), distance.
    """
    collection = _get_collection()

    if collection.count() == 0:
        return []

    # Clamp n_results to collection size
    actual_n = min(n_results, collection.count())

    results = collection.query(
        query_texts=[query],
        n_results=actual_n,
        include=["documents", "metadatas", "distances"],
    )

    if not results["documents"] or not results["documents"][0]:
        return []

    matches = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        matches.append({
            "title_en": meta.get("title_en", ""),
            "title_ro": meta.get("title_ro", ""),
            "url": meta.get("url", ""),
            "timestamp": meta.get("timestamp", ""),
            "body_en": doc,
            "distance": dist,
        })

    return matches


def search_biziday_formatted(query: str, n_results: int = BIZIDAY_SEARCH_RESULTS) -> str:
    """Search and return results as a formatted string for LLM consumption."""
    matches = search_biziday(query, n_results)

    if not matches:
        return "No Biziday news articles found matching the query."

    parts = []
    for i, m in enumerate(matches, 1):
        part = f"[Article {i}] {m['title_en']}\n"
        part += f"URL: {m['url']}\n"
        part += f"Date: {m['timestamp']}\n"
        # Include the article body, truncated for context window
        body = m["body_en"]
        if len(body) > 4000:
            body = body[:4000] + "\n[…truncated]"
        part += f"Content:\n{body}"
        parts.append(part)

    return (
        f'Biziday news search results for: "{query}"\n\n'
        + "\n\n".join(parts)
        + "\n\nUse the articles above to answer the user's question about Romanian news."
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Collection Management
# ═══════════════════════════════════════════════════════════════════════════

def clear_collection() -> int:
    """Delete all documents from the Biziday collection. Returns count removed."""
    collection = _get_collection()
    count = collection.count()
    if count > 0:
        persist_dir = os.path.expanduser(CACHE_DIR)
        client = chromadb.PersistentClient(path=persist_dir)
        client.delete_collection(BIZIDAY_COLLECTION_NAME)
        # Recreate empty collection
        client.get_or_create_collection(
            name=BIZIDAY_COLLECTION_NAME,
            metadata={"hnsw:space": "l2"},
        )
    return count


def collection_stats() -> dict:
    """Return stats about the Biziday news collection."""
    collection = _get_collection()
    return {
        "collection_name": BIZIDAY_COLLECTION_NAME,
        "document_count": collection.count(),
        "persist_dir": os.path.expanduser(CACHE_DIR),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Index Biziday news articles into ChromaDB with English translation"
    )
    parser.add_argument(
        "--input", "-i",
        default=DEFAULT_JSON_PATH,
        help=f"Input JSON file path (default: {DEFAULT_JSON_PATH})",
    )
    parser.add_argument(
        "--search", "-s",
        type=str,
        default=None,
        help="Search indexed articles by query",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear all indexed articles",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show collection statistics",
    )
    args = parser.parse_args()

    # ── Stats ────────────────────────────────────────────────────────────
    if args.stats:
        stats = collection_stats()
        print(f"\n{CYAN}📊  Biziday News Collection Stats{RESET}")
        print(f"   Collection : {stats['collection_name']}")
        print(f"   Documents  : {stats['document_count']}")
        print(f"   Storage    : {stats['persist_dir']}\n")
        return

    # ── Clear ────────────────────────────────────────────────────────────
    if args.clear:
        removed = clear_collection()
        print(f"\n{YELLOW}🗑  Cleared {removed} articles from '{BIZIDAY_COLLECTION_NAME}'{RESET}\n")
        return

    # ── Search ───────────────────────────────────────────────────────────
    if args.search:
        print(f"\n{CYAN}🔍  Searching for: \"{args.search}\"{RESET}\n")
        matches = search_biziday(args.search)
        if not matches:
            print(f"  {YELLOW}No matching articles found.{RESET}\n")
            return

        for i, m in enumerate(matches, 1):
            print(f"  {BOLD}[{i}]{RESET} {m['title_en']}")
            print(f"      {DIM}RO: {m['title_ro'][:80]}...{RESET}")
            print(f"      {DIM}URL: {m['url']}{RESET}")
            print(f"      {DIM}Date: {m['timestamp']}  |  Distance: {m['distance']:.4f}{RESET}")
            # Show a short preview of the body
            body_preview = m["body_en"][:200].replace("\n", " ")
            print(f"      {DIM}{body_preview}...{RESET}\n")
        return

    # ── Index ────────────────────────────────────────────────────────────
    index_articles(json_path=args.input)


if __name__ == "__main__":
    main()
