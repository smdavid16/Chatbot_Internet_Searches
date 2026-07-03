"""
Internet Search Chatbot
=======================
A CLI chatbot powered by Ollama (Qwen2.5 3B) with live Google search
capabilities via tool calling.

The model decides *when* to search — only invoking the tools when it
needs up-to-date or factual information it doesn't already know.
"""

import re
import sys
import json
import textwrap
from datetime import datetime, timezone
from zoneinfo import ZoneInfo, available_timezones

import ntplib
import ollama
import requests
from bs4 import BeautifulSoup
from urllib.parse import unquote

from config import (
    MODEL_NAME,
    NUM_SEARCH_RESULTS,
    SEARXNG_URL,
    OPENWEATHERMAP_API_KEY,
    MAX_PAGE_CONTENT_LENGTH,
    PAGE_REQUEST_TIMEOUT,
    MAX_TOOL_ITERATIONS,
    CACHE_DIR,
    CACHE_TTL_SECONDS,
    CACHE_SIMILARITY_THRESHOLD,
)
from search_cache import SearchCache

# ── Colours for the terminal ────────────────────────────────────────────────
CYAN    = "\033[96m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
RED     = "\033[91m"
DIM     = "\033[2m"
BOLD    = "\033[1m"
RESET   = "\033[0m"

# ── Search Cache ────────────────────────────────────────────────────────────
import os as _os
_cache = SearchCache(
    persist_dir=_os.path.expanduser(CACHE_DIR),
    ttl_seconds=CACHE_TTL_SECONDS,
    similarity_threshold=CACHE_SIMILARITY_THRESHOLD,
)

SYSTEM_PROMPT = textwrap.dedent("""\
    You are a helpful assistant. You have tools to search the internet.

    RULE 1: Your training data is OUTDATED. You MUST use google_search
    to find current facts. NEVER answer factual questions from memory.

    RULE 2: When the user asks about ANY person, leader, president,
    CEO, event, score, price, or anything that can change — you MUST
    call google_search FIRST, BEFORE answering.

    RULE 3: After google_search returns results, read the KEY FACTS
    carefully. The answer is there. Use those facts in your response.
    Do NOT ignore the search results.

    RULE 4: Always include the source URL in your answer.

    RULE 5: Use get_current_datetime ONLY when the user asks "what
    time is it" or "what is today's date". Do NOT use it to answer
    factual questions about people or events.

    RULE 6: Use get_current_weather ONLY when the user asks about the weather.

    RULE 7: If you are not sure, SEARCH. When in doubt, SEARCH.
""")


# ═══════════════════════════════════════════════════════════════════════════
#  Tool Functions
#  ──────────────
#  Ollama automatically converts these into JSON-schema tool definitions
#  by inspecting the function signature, type hints, and docstring.
# ═══════════════════════════════════════════════════════════════════════════

def google_search(query: str) -> str:
    """Search the internet for the given query and return the top results
    with key facts extracted from each page.  Use this tool whenever
    you need up-to-date information from the internet.  The answer
    to the user's question is in the KEY FACTS section."""

    # ── Check cache first ─────────────────────────────────────────────
    cached = _cache.lookup(query)
    if cached:
        print(f"\n  {GREEN}💾  Cache hit! Returning saved results for similar query.{RESET}")
        return f"[Cached result]\n{cached}"

    print(f"\n  {DIM}🌐  Fetching fresh results…{RESET}")
    print(f"  {DIM}🔍  Searching for: \"{query}\"{RESET}")

    results = _searxng_search(query)

    if not results:
        return "No search results found."

    # Fetch and extract relevant snippets from each page
    print(f"  {DIM}📄  Fetching & analysing {len(results)} page(s)…{RESET}")
    output_parts = []
    for i, r in enumerate(results, 1):
        page_text = _fetch_page(r["url"])
        snippets = _extract_relevant_snippets(page_text, query) if page_text else ""

        part = f"[Source {i}] {r['title']}\n"
        part += f"URL: {r['url']}\n"
        if r.get("snippet"):
            part += f"Description: {r['snippet']}\n"
        if snippets:
            part += f"KEY FACTS:\n{snippets}"
        else:
            part += "KEY FACTS: (could not extract content from this page)"
        output_parts.append(part)

    output = (
        f"Search results for: \"{query}\"\n\n"
        + "\n\n".join(output_parts)
        + "\n\nUse the KEY FACTS above to answer the user's question."
    )

    # ── Store in cache ────────────────────────────────────────────────
    _cache.store(query, output)
    print(f"  {DIM}💾  Results cached for future queries.{RESET}")

    return output


def _searxng_search(query: str, num_results: int | None = None) -> list[dict]:
    """Search SearXNG via its JSON endpoint and parse the results.
    Returns a list of dicts with 'title', 'url', and 'snippet' keys."""
    if num_results is None:
        num_results = NUM_SEARCH_RESULTS

    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
        }
        params = {
            "q": query,
            "format": "json"
        }
        resp = requests.get(
            f"{SEARXNG_URL.rstrip('/')}/search",
            headers=headers, params=params, timeout=PAGE_REQUEST_TIMEOUT,
        )
        resp.raise_for_status()

        data = resp.json()
        results = []

        for item in data.get("results", [])[:num_results]:
            title = item.get("title", "")
            url = item.get("url", "")
            snippet = item.get("content", "")

            results.append({
                "title": title,
                "url": url,
                "snippet": snippet,
            })

        return results

    except Exception as exc:
        print(f"  {RED}✖  Search error: {exc}{RESET}")
        return []


def scrape_webpage(url: str, query: str = "") -> str:
    """Fetch and read a specific web page URL to find detailed information.
    Use this tool when you need more detail from a URL found in search results.

    Args:
        url: The full URL of the web page to read.
        query: What you are looking for on this page (helps extract relevant content).
    """

    print(f"\n  {DIM}📄  Scraping page: {url}{RESET}")

    page_text = _fetch_page(url, max_length=MAX_PAGE_CONTENT_LENGTH * 3)
    if not page_text:
        return f"Could not fetch content from {url}"

    if query:
        snippets = _extract_relevant_snippets(page_text, query, max_snippets=20)
        if snippets:
            return f"=== Relevant content from {url} ===\n{snippets}"

    # Fallback: return truncated full text
    if len(page_text) > MAX_PAGE_CONTENT_LENGTH:
        page_text = page_text[:MAX_PAGE_CONTENT_LENGTH] + "\n[…truncated]"
    return f"=== Content from {url} ===\n{page_text}"


def get_current_datetime(timezone_name: str = "UTC") -> str:
    """Get the current date and time from an NTP (Network Time Protocol)
    server.  Use this tool whenever the user asks for the current time,
    date, day of the week, or any time-related question.

    Args:
        timezone_name: An IANA timezone name such as 'America/New_York',
            'Europe/London', 'Asia/Tokyo', 'Australia/Sydney', or 'UTC'.
            Use this to return the time in a specific region of the world.
            Defaults to 'UTC' if not specified.
    """

    print(f"\n  {DIM}🕐  Querying NTP server for accurate time (tz: {timezone_name})…{RESET}")

    # Validate the timezone name
    try:
        tz = ZoneInfo(timezone_name)
    except (KeyError, Exception):
        # Try a case-insensitive fuzzy match before giving up
        match = _fuzzy_match_timezone(timezone_name)
        if match:
            tz = ZoneInfo(match)
            timezone_name = match
        else:
            available = ", ".join(sorted(available_timezones())[:30])
            return (
                f"Unknown timezone: '{timezone_name}'.\n"
                f"Use an IANA timezone name, e.g.: America/New_York, "
                f"Europe/London, Asia/Tokyo, Australia/Sydney, UTC.\n"
                f"Some valid options: {available}…"
            )

    try:
        client = ntplib.NTPClient()
        response = client.request("pool.ntp.org", version=3)
        utc_time = datetime.fromtimestamp(response.tx_time, tz=timezone.utc)
    except Exception as exc:
        return f"NTP request failed: {exc}"

    # Convert to the requested timezone
    local_time = utc_time.astimezone(tz)

    # Format in several useful ways
    iso_str = local_time.strftime("%Y-%m-%dT%H:%M:%S %Z")
    human_str = local_time.strftime("%A, %B %d, %Y at %H:%M:%S %Z")
    utc_offset = local_time.strftime("%z")

    return (
        f"Current date/time in {timezone_name} (UTC{utc_offset}):\n"
        f"  ISO 8601 : {iso_str}\n"
        f"  Readable : {human_str}\n"
        f"  UTC time : {utc_time.strftime('%Y-%m-%dT%H:%M:%S UTC')}"
    )


def get_current_weather(location: str) -> str:
    """Get the current weather for a specific location. Use this tool whenever
    the user asks for the weather in a specific city, state, or country.

    Args:
        location: The name of the city/location (e.g., 'London', 'New York, US', 'Tokyo').
    """
    if not OPENWEATHERMAP_API_KEY:
        return "Weather tool is disabled: OpenWeatherMap API key is missing from config."

    print(f"\n  {DIM}⛅  Fetching current weather for '{location}'…{RESET}")

    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": location,
        "appid": OPENWEATHERMAP_API_KEY,
        "units": "metric"
    }

    try:
        response = requests.get(url, params=params, timeout=PAGE_REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()

        weather_desc = data["weather"][0]["description"].capitalize()
        temp = data["main"]["temp"]
        feels_like = data["main"]["feels_like"]
        humidity = data["main"]["humidity"]
        wind_speed = data["wind"]["speed"]
        city = data["name"]
        country = data["sys"].get("country", "")

        return (
            f"Current weather in {city}, {country}:\n"
            f"  Condition : {weather_desc}\n"
            f"  Temp      : {temp}°C (feels like {feels_like}°C)\n"
            f"  Humidity  : {humidity}%\n"
            f"  Wind      : {wind_speed} m/s"
        )
    except requests.exceptions.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            return f"Weather data not found for location: '{location}'"
        if exc.response is not None and exc.response.status_code == 401:
            return "Weather tool is disabled: Invalid OpenWeatherMap API key."
        return f"Could not fetch weather: {exc}"
    except Exception as exc:
        return f"Could not fetch weather: {exc}"


def _fuzzy_match_timezone(name: str) -> str | None:
    """Try to find a matching IANA timezone by case-insensitive search
    on the city part (e.g. 'tokyo' -> 'Asia/Tokyo')."""
    name_lower = name.lower().replace(" ", "_")
    for tz in sorted(available_timezones()):
        if tz.lower() == name_lower:
            return tz
        # Match on just the city part (after the last /)
        city = tz.rsplit("/", 1)[-1].lower()
        if city == name_lower:
            return tz
    return None


def _fetch_page(url: str, max_length: int | None = None) -> str:
    """Fetch a web page and extract its main text content.
    Prioritises article/main content areas to reduce noise."""
    if max_length is None:
        max_length = MAX_PAGE_CONTENT_LENGTH

    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
        }
        response = requests.get(
            url, headers=headers, timeout=PAGE_REQUEST_TIMEOUT
        )
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # ── Extract metadata summary ──────────────────────────────────
        summary_parts = []

        # Page title
        if soup.title and soup.title.string:
            summary_parts.append(f"Page Title: {soup.title.string.strip()}")

        # Meta description
        meta_desc = soup.find("meta", attrs={"name": "description"})
        if meta_desc and meta_desc.get("content"):
            summary_parts.append(f"Description: {meta_desc['content'].strip()}")

        # og:description (often more detailed)
        og_desc = soup.find("meta", attrs={"property": "og:description"})
        if og_desc and og_desc.get("content"):
            summary_parts.append(f"Summary: {og_desc['content'].strip()}")

        # Headings (h1, h2) — often contain key facts
        headings = []
        for tag in soup.find_all(["h1", "h2"], limit=10):
            heading_text = tag.get_text(strip=True)
            if heading_text and len(heading_text) < 200:
                headings.append(f"  • {heading_text}")
        if headings:
            summary_parts.append("Key Headings:\n" + "\n".join(headings))

        # ── Remove non-content elements ───────────────────────────────
        for tag in soup(["script", "style", "nav", "header", "footer",
                         "aside", "form", "noscript", "iframe", "svg",
                         "button", "input", "select", "textarea",
                         "menu", "menuitem"]):
            tag.decompose()

        # ── Try to find the main content area ─────────────────────────
        main_content = None

        # Priority 1: <article> tag
        article = soup.find("article")
        if article:
            main_content = article

        # Priority 2: <main> tag
        if not main_content:
            main_tag = soup.find("main")
            if main_tag:
                main_content = main_tag

        # Priority 3: Common content div IDs/classes
        if not main_content:
            for selector in [
                {"id": "content"}, {"id": "main-content"},
                {"id": "mw-content-text"},  # Wikipedia
                {"class_": "post-content"}, {"class_": "article-body"},
                {"class_": "entry-content"}, {"class_": "story-body"},
                {"role": "main"},
            ]:
                found = soup.find("div", **selector)
                if not found:
                    found = soup.find("section", **selector)
                if found:
                    main_content = found
                    break

        # Priority 4: Largest <div> by text length (heuristic)
        if not main_content:
            body = soup.find("body")
            if body:
                divs = body.find_all("div", recursive=False)
                if divs:
                    main_content = max(divs, key=lambda d: len(d.get_text()))

        # Fallback: use entire body
        source = main_content or soup.find("body") or soup

        text = source.get_text(separator="\n", strip=True)

        # Collapse excessive whitespace
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        # Remove very short lines that are likely navigation remnants
        lines = [l for l in lines if len(l) > 2]
        text = "\n".join(lines)

        # Prepend the summary
        if summary_parts:
            summary = "\n".join(summary_parts)
            text = f"{summary}\n\n--- Main Content ---\n{text}"

        # Truncate to stay within context limits
        if len(text) > max_length:
            text = text[:max_length] + "\n[…truncated]"

        return text

    except Exception as exc:
        return f"[Could not fetch page: {exc}]"


def _extract_relevant_snippets(
    text: str, query: str, max_snippets: int = 10
) -> str:
    """Score sentences by relevance to the query and return the top ones.
    This pre-digests page content so the small model doesn't have to
    search through thousands of characters of noise."""

    if not text or not query:
        return text[:2000] if text else ""

    # Build keyword set from the query (lowercase, skip short/stop words)
    stop_words = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "of", "in", "to", "for",
        "with", "on", "at", "by", "from", "as", "into", "about", "between",
        "through", "and", "but", "or", "not", "no", "if", "that", "this",
        "it", "what", "who", "whom", "which", "when", "where", "how", "why",
        "me", "my", "i", "you", "your", "he", "she", "they", "we", "our",
        "its", "his", "her", "their",
    }
    query_words = {
        w for w in re.findall(r"[a-z]+", query.lower())
        if len(w) > 2 and w not in stop_words
    }

    if not query_words:
        # Fallback if query is all stop words
        return text[:2000]

    # Split text into sentences
    sentences = re.split(r'(?<=[.!?])\s+|\n', text)
    # Remove empty / very short fragments
    sentences = [s.strip() for s in sentences if len(s.strip()) > 15]

    # Score each sentence
    scored = []
    for sent in sentences:
        sent_lower = sent.lower()
        sent_words = set(re.findall(r"[a-z]+", sent_lower))
        # Count how many query keywords appear in this sentence
        hits = len(query_words & sent_words)
        if hits > 0:
            # Bonus for sentences that contain multiple keywords together
            score = hits + (0.5 if hits >= 2 else 0)
            scored.append((score, sent))

    if not scored:
        # No keyword matches — return the first chunk of text as fallback
        return text[:2000]

    # Sort by score descending, take top N
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:max_snippets]

    # Format as bullet points for clarity
    snippets = []
    for _score, sent in top:
        # Clean up the sentence
        sent = sent.strip()
        if not sent.endswith(('.', '!', '?', ':', '"', "'")):
            sent += '...'  # indicate it was cut
        snippets.append(f"  • {sent}")

    return "\n".join(snippets)


# ═══════════════════════════════════════════════════════════════════════════
#  Agent Loop
# ═══════════════════════════════════════════════════════════════════════════

# Map tool names to their Python callables
TOOL_REGISTRY = {
    "google_search": google_search,
    "scrape_webpage": scrape_webpage,
    "get_current_datetime": get_current_datetime,
    "get_current_weather": get_current_weather,
}

# The list of tools Ollama will advertise to the model
TOOLS = [google_search, scrape_webpage, get_current_datetime, get_current_weather]


# Keywords that suggest the user is asking a factual question that
# requires searching (used by the search-enforcement logic below).
_FACTUAL_KEYWORDS = re.compile(
    r'\b(who|what|where|when|president|prime minister|leader|ceo|king|queen|'
    r'chancellor|governor|mayor|current|latest|recent|today|now|score|price|'
    r'weather|population|capital|founded|born|died|age|height|worth|salary|'
    r'winner|champion|election|result|stock|rate|cost|how many|how much)\b',
    re.IGNORECASE,
)


def agent_turn(messages: list[dict]) -> str:
    """Run a single agent turn: call the model, execute any tool calls,
    and loop until the model produces a final text response."""

    used_tool = False

    for iteration in range(MAX_TOOL_ITERATIONS):
        response = ollama.chat(
            model=MODEL_NAME,
            messages=messages,
            tools=TOOLS,
        )

        assistant_msg = response.message

        # If no tool calls, we have our final answer — but check if the
        # model skipped searching for a factual question.
        if not assistant_msg.tool_calls:
            if not used_tool and iteration < 2:
                # Check if the user's last message looks factual
                user_msg = ""
                for m in reversed(messages):
                    if m.get("role") == "user" or (
                        hasattr(m, "role") and m.role == "user"
                    ):
                        user_msg = m.get("content", "") if isinstance(m, dict) else getattr(m, "content", "")
                        break

                if user_msg and _FACTUAL_KEYWORDS.search(user_msg):
                    print(f"  {DIM}🔄  Nudging model to use tools first…{RESET}")
                    # Don't append the non-search answer; instead nudge
                    messages.append({
                        "role": "user",
                        "content": (
                            "You must use a tool (like google_search or get_current_datetime) to answer this "
                            "question. Do NOT answer from memory. Call a tool now."
                        ),
                    })
                    continue

            return assistant_msg.content

        # Process each tool call
        messages.append(assistant_msg)
        used_tool = True

        for tool_call in assistant_msg.tool_calls:
            fn_name = tool_call.function.name
            fn_args = tool_call.function.arguments

            print(f"  {YELLOW}⚙  Tool call: {fn_name}({json.dumps(fn_args)}){RESET}")

            func = TOOL_REGISTRY.get(fn_name)
            if func:
                try:
                    result = func(**fn_args)
                except Exception as exc:
                    result = f"Tool execution error: {exc}"
            else:
                result = f"Unknown tool: {fn_name}"

            # Feed the result back to the model
            messages.append({
                "role": "tool",
                "content": str(result),
            })

    return (
        "I wasn't able to complete my research within the allowed number "
        "of steps. Please try rephrasing your question."
    )


# ═══════════════════════════════════════════════════════════════════════════
#  Main Conversation Loop
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print(f"\n{BOLD}{CYAN}╔══════════════════════════════════════════════╗{RESET}")
    print(f"{BOLD}{CYAN}║   🌐  Internet Search Chatbot               ║{RESET}")
    print(f"{BOLD}{CYAN}║   Model: {MODEL_NAME:<36}║{RESET}")
    print(f"{BOLD}{CYAN}╚══════════════════════════════════════════════╝{RESET}")
    print(f"{DIM}Type your message and press Enter. Type 'quit' or 'exit' to leave.{RESET}\n")

    # Conversation history persists across turns for memory
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    while True:
        try:
            user_input = input(f"{GREEN}{BOLD}You: {RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{DIM}Goodbye!{RESET}")
            sys.exit(0)

        if not user_input:
            continue

        if user_input.lower() in ("quit", "exit", "q"):
            print(f"{DIM}Goodbye!{RESET}")
            break

        # ── Slash commands ────────────────────────────────────────────
        if user_input.lower() == "/clear-cache":
            removed = _cache.clear()
            print(f"{YELLOW}🗑  Cache cleared ({removed} entries removed).{RESET}\n")
            continue

        if user_input.lower() == "/cache-stats":
            stats = _cache.stats()
            print(f"{CYAN}💾  Cache stats:{RESET}")
            print(f"   Entries: {stats['entries']}")
            print(f"   TTL: {stats['ttl_seconds'] // 3600}h")
            print(f"   Similarity threshold: {stats['similarity_threshold']}\n")
            continue

        messages.append({"role": "user", "content": user_input})

        print(f"\n{DIM}Thinking…{RESET}")

        try:
            answer = agent_turn(messages)
        except Exception as exc:
            answer = f"An error occurred: {exc}"
            print(f"  {RED}✖  {answer}{RESET}")

        messages.append({"role": "assistant", "content": answer})

        print(f"\n{CYAN}{BOLD}Assistant:{RESET} {answer}\n")


if __name__ == "__main__":
    main()
