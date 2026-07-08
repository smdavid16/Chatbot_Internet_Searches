"""
Clean Biziday JSON
==================
Post-processes the stiriBiziday.json file produced by the UiPath scraper.
Fetches each article page and extracts only the meaningful text content
from <p>, <h1>, <h2>, <h3> elements inside the article body.

Usage:
    python clean_biziday.py
    python clean_biziday.py path/to/stiriBiziday.json
"""

import json
import sys
import os

# Fix console encoding for Romanian characters
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import requests
from bs4 import BeautifulSoup


# Default paths — checks both UiPath project and chatbot project locations
DEFAULT_PATHS = [
    os.path.join(os.path.dirname(__file__), "stiriBiziday.json"),
    os.path.expanduser(r"~\Documents\UiPath\Biziday\stiriBiziday.json"),
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
}


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
        # Fallback: try single-content wrapper
        content_div = soup.find("div", class_="single-content")
    if not content_div:
        # Last fallback: use <article> tag
        content_div = soup.find("article")
    if not content_div:
        content_div = soup

    # Extract text from only meaningful elements
    parts = []
    for el in content_div.find_all(["h1", "h2", "h3", "p"]):
        text = el.get_text(strip=True)
        if text and len(text) > 2:
            # Prefix headings for readability
            if el.name in ("h1", "h2", "h3"):
                parts.append(f"[{el.name.upper()}] {text}")
            else:
                parts.append(text)

    if not parts:
        return "[No article content found]"

    return "\n\n".join(parts)


def clean_json(filepath: str) -> None:
    """Read the JSON, clean each article's HTML_Sursa, and save back."""
    print(f"Reading: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        articles = json.load(f)

    print(f"Found {len(articles)} articles. Fetching clean content...\n")

    for i, article in enumerate(articles):
        url = article.get("URL", "")
        title = article.get("Titlu", "")[:80]
        print(f"  [{i+1}/{len(articles)}] {title}...")

        if url:
            clean_text = extract_article_text(url)
            article["HTML_Sursa"] = clean_text
            # Show a preview
            preview = clean_text[:120].replace("\n", " ")
            print(f"           → {preview}...")
        else:
            print(f"           → No URL, skipping")

    # Write back
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)

    print(f"\n✓ Cleaned JSON saved to: {filepath}")


def main():
    # Determine file path
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = None
        for path in DEFAULT_PATHS:
            if os.path.exists(path):
                filepath = path
                break

    if not filepath or not os.path.exists(filepath):
        print("Error: stiriBiziday.json not found.")
        print(f"Searched: {DEFAULT_PATHS}")
        print("Usage: python clean_biziday.py [path/to/stiriBiziday.json]")
        sys.exit(1)

    clean_json(filepath)


if __name__ == "__main__":
    main()
