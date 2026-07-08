""" python scrape_biziday.py
    python scrape_biziday.py --output /path/to/output.json
    python scrape_biziday.py --count 10
"""

import json
import sys
import os
import argparse
from datetime import datetime

import requests
from bs4 import BeautifulSoup

# Fix console encoding for Romanian characters (Windows only)
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BIZIDAY_URL = "https://www.biziday.ro/"
DEFAULT_OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stiriBiziday.json")
DEFAULT_COUNT = 20

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ro-RO,ro;q=0.9,en-US;q=0.8,en;q=0.7",
}


def scrape_homepage(count: int = DEFAULT_COUNT) -> list[dict]:
    """Scrape the Biziday homepage and return article headlines + URLs."""
    print(f"🌐 Fetching Biziday homepage...")

    resp = requests.get(BIZIDAY_URL, headers=HEADERS, timeout=15)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    # Biziday structure: <ul class="loop"> → <li class="article"> → <a class="post-url">
    articles = []
    article_items = soup.select("ul.loop li.article:not(.is-ad)")

    for li in article_items[:count]:
        link = li.select_one("a.post-url")
        if not link:
            continue

        url = link.get("href", "")
        if url and not url.startswith("http"):
            url = f"https://www.biziday.ro{url}"

        headline_el = li.select_one("h2.post-title span[itemprop='headline']")
        if not headline_el:
            headline_el = li.select_one("h2.post-title")

        headline = headline_el.get_text(strip=True) if headline_el else ""

        if url and headline:
            articles.append({"headline": headline, "url": url})

    print(f"   Found {len(articles)} articles")
    return articles


def extract_article_text(url: str) -> str:
    """Fetch an article page and extract clean text from p, h1, h2, h3 elements."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
    except Exception as exc:
        return f"[Error fetching page: {exc}]"

    soup = BeautifulSoup(resp.text, "html.parser")

    # Target the article content div
    content_div = soup.find("div", class_="post-content")
    if not content_div:
        content_div = soup.find("div", class_="single-content")
    if not content_div:
        content_div = soup.find("article")
    if not content_div:
        content_div = soup

    # Extract text from only meaningful elements
    parts = []
    for el in content_div.find_all(["h1", "h2", "h3", "p"]):
        text = el.get_text(strip=True)
        if text and len(text) > 2:
            if el.name in ("h1", "h2", "h3"):
                parts.append(f"[{el.name.upper()}] {text}")
            else:
                parts.append(text)

    if not parts:
        return "[No article content found]"

    return "\n\n".join(parts)


def scrape_biziday(count: int = DEFAULT_COUNT, output_path: str = DEFAULT_OUTPUT) -> None:
    """Full scraping pipeline: homepage → articles → clean JSON."""
    # Step 1: Get article list from homepage
    article_list = scrape_homepage(count)

    if not article_list:
        print("❌ No articles found on the homepage.")
        sys.exit(1)

    # Step 2: Fetch and extract each article
    results = []
    for i, item in enumerate(article_list):
        title_preview = item["headline"][:80]
        print(f"  [{i+1}/{len(article_list)}] {title_preview}...")

        article_text = extract_article_text(item["url"])
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        results.append({
            "Titlu": item["headline"],
            "HTML_Sursa": article_text,
            "URL": item["url"],
            "Timestamp": timestamp,
        })

        # Show preview
        preview = article_text[:100].replace("\n", " ")
        print(f"           → {preview}...")

    # Step 3: Save JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Saved {len(results)} articles to: {output_path}")
    size_kb = os.path.getsize(output_path) / 1024
    print(f"   File size: {size_kb:.1f} KB")


def main():
    parser = argparse.ArgumentParser(description="Scrape Biziday.ro news articles")
    parser.add_argument(
        "--output", "-o",
        default=DEFAULT_OUTPUT,
        help=f"Output JSON file path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--count", "-n",
        type=int,
        default=DEFAULT_COUNT,
        help=f"Number of articles to scrape (default: {DEFAULT_COUNT})",
    )
    parser.add_argument(
        "--index",
        action="store_true",
        help="Translate and index scraped articles into ChromaDB after saving JSON",
    )
    args = parser.parse_args()

    scrape_biziday(count=args.count, output_path=args.output)

    if args.index:
        from index_biziday import index_articles
        index_articles(json_path=args.output)


if __name__ == "__main__":
    main()
