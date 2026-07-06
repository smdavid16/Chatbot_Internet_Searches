"""
SeamlessM4T Translation Module
==============================
Provides Romanian ↔ English text translation using Meta's SeamlessM4T model.

Language detection is automatic:
  - If the input looks Romanian → translate to English
  - If the input looks English  → pass through unchanged

The module tries to load the large v2 model (2.3B) first.  If that fails
due to insufficient VRAM, it falls back to the medium model (1.2B).
"""

import re
# pyrefly: ignore [missing-import]
import torch
# pyrefly: ignore [missing-import]
from transformers import AutoProcessor, SeamlessM4Tv2Model, SeamlessM4TModel

from config import SEAMLESS_MODEL_LARGE, SEAMLESS_MODEL_FALLBACK

# ── Terminal colours (keep in sync with chatbot.py) ─────────────────────────
DIM   = "\033[2m"
GREEN = "\033[92m"
RED   = "\033[91m"
YELLOW = "\033[93m"
CYAN  = "\033[96m"
RESET = "\033[0m"

# ── Singleton state ─────────────────────────────────────────────────────────
_processor = None
_model = None
_device = None
_loaded_model_name = None


# ═══════════════════════════════════════════════════════════════════════════
#  Model Loading
# ═══════════════════════════════════════════════════════════════════════════

def _load_model():
    """Load the SeamlessM4T model and processor into GPU (or CPU fallback).

    Tries the large v2 model first; falls back to the medium model if
    CUDA runs out of memory.
    """
    global _processor, _model, _device, _loaded_model_name

    _device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Attempt 1: Large v2 (2.3B) ──────────────────────────────────────
    print(f"\n  {DIM}🌍  Loading translation model: {SEAMLESS_MODEL_LARGE}…{RESET}")
    try:
        _processor = AutoProcessor.from_pretrained(
            SEAMLESS_MODEL_LARGE,
            clean_up_tokenization_spaces=False,
        )
        _model = SeamlessM4Tv2Model.from_pretrained(
            SEAMLESS_MODEL_LARGE,
            torch_dtype=torch.float16 if _device == "cuda" else torch.float32,
        ).to(_device)
        _loaded_model_name = SEAMLESS_MODEL_LARGE
        print(f"  {GREEN}✔  Translation model loaded: {SEAMLESS_MODEL_LARGE} "
              f"on {_device.upper()}{RESET}\n")
        return
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        print(f"  {YELLOW}⚠  Large model failed ({exc.__class__.__name__}), "
              f"falling back to medium model…{RESET}")
        # Free any partially-loaded tensors
        _model = None
        _processor = None
        if _device == "cuda":
            torch.cuda.empty_cache()

    # ── Attempt 2: Medium (1.2B) ────────────────────────────────────────
    print(f"  {DIM}🌍  Loading fallback model: {SEAMLESS_MODEL_FALLBACK}…{RESET}")
    try:
        _processor = AutoProcessor.from_pretrained(
            SEAMLESS_MODEL_FALLBACK,
            clean_up_tokenization_spaces=False,
        )
        _model = SeamlessM4TModel.from_pretrained(
            SEAMLESS_MODEL_FALLBACK,
            torch_dtype=torch.float16 if _device == "cuda" else torch.float32,
        ).to(_device)
        _loaded_model_name = SEAMLESS_MODEL_FALLBACK
        print(f"  {GREEN}✔  Translation model loaded: {SEAMLESS_MODEL_FALLBACK} "
              f"on {_device.upper()}{RESET}\n")
    except Exception as exc:
        print(f"  {RED}✖  Could not load any translation model: {exc}{RESET}")
        print(f"  {RED}   Translation will be DISABLED for this session.{RESET}\n")
        _model = None
        _processor = None
        _loaded_model_name = None


def _ensure_model():
    """Lazy-load the model on first use."""
    if _model is None and _loaded_model_name is None:
        _load_model()


def get_loaded_model_name() -> str | None:
    """Return the name of the currently loaded translation model, or None."""
    _ensure_model()
    return _loaded_model_name


# ═══════════════════════════════════════════════════════════════════════════
#  Language Detection
# ═══════════════════════════════════════════════════════════════════════════

# Romanian-specific diacritics
_RO_DIACRITICS = set("ăâîșțĂÂÎȘȚ")

# Common Romanian words that are unlikely in English
_RO_MARKERS = re.compile(
    r'\b(este|sunt|care|pentru|despre|poate|acum|unde|cine|când|cum|într|'
    r'într-un|într-o|acest|această|acestea|aceștia|decât|dar|sau|ori|'
    r'foarte|bine|mai|după|prin|dintre|doar|încă|aici|acolo|'
    r'președintele|guvernul|România|român[ăi]?|moldov[ae]|București|'
    r'ce|nu|da|și|la|din|pe|cu|de|un|o|al|ai|ale|că|ca|'
    r'eu|tu|el|ea|noi|voi|ei|ele|'
    r'vreau|vrei|vrea|avem|aveți|au|'
    r'știi|știe|știm|fac|faci|face|facem|'
    r'bună|salut|mulțumesc|te rog)\b',
    re.IGNORECASE | re.UNICODE,
)


def detect_language(text: str) -> str:
    """Detect whether *text* is Romanian or English.

    Returns ``"ron"`` for Romanian, ``"eng"`` for English.

    Uses a fast heuristic approach:
      1. Romanian diacritics (ă, â, î, ș, ț) → strong Romanian signal
      2. Romanian marker words → moderate signal
      3. Default to English if nothing triggers
    """
    if not text or not text.strip():
        return "eng"

    # Check for Romanian diacritics
    has_diacritics = any(ch in _RO_DIACRITICS for ch in text)

    # Count Romanian marker-word hits
    ro_hits = len(_RO_MARKERS.findall(text))

    # Word count for ratio calculation
    words = text.split()
    word_count = max(len(words), 1)

    # Decision logic
    if has_diacritics:
        return "ron"

    # If ≥15% of words match Romanian markers, call it Romanian
    if ro_hits / word_count >= 0.15:
        return "ron"

    return "eng"


# ═══════════════════════════════════════════════════════════════════════════
#  Translation Functions
# ═══════════════════════════════════════════════════════════════════════════

def _translate(text: str, src_lang: str, tgt_lang: str) -> str:
    """Translate *text* from *src_lang* to *tgt_lang* using SeamlessM4T.

    Returns the original text unchanged if the model is not loaded.
    """
    _ensure_model()

    if _model is None or _processor is None:
        return text  # no model available

    try:
        inputs = _processor(text=text, src_lang=src_lang, return_tensors="pt")
        # Move input tensors to the same device as the model
        inputs = {k: v.to(_device) if hasattr(v, "to") else v
                  for k, v in inputs.items()}

        output_tokens = _model.generate(
            **inputs,
            tgt_lang=tgt_lang,
            generate_speech=False,
            max_new_tokens=512,
        )

        translated = _processor.decode(
            output_tokens[0].tolist()[0],
            skip_special_tokens=True,
        )
        return translated.strip()

    except Exception as exc:
        print(f"  {RED}✖  Translation error: {exc}{RESET}")
        return text  # return original on failure


# ── Pre-processing helpers for robust translation ───────────────────────────

_MAX_CHUNK_CHARS = 800   # max characters per chunk sent to the model


def _strip_markdown(text: str) -> str:
    """Remove markdown formatting to produce clean text for translation.

    Preserves paragraph structure (newlines) but strips syntax markers
    that confuse the translation model (bold, italic, links, headings,
    code backticks, etc.).
    """
    # Bold-italic (***text***) first, then bold, then italic
    text = re.sub(r'\*{3}(.*?)\*{3}', r'\1', text)
    text = re.sub(r'\*{2}(.*?)\*{2}', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'_{3}(.*?)_{3}', r'\1', text)
    text = re.sub(r'_{2}(.*?)_{2}', r'\1', text)
    text = re.sub(r'(?<!\w)_(.+?)_(?!\w)', r'\1', text)

    # Inline code backticks
    text = re.sub(r'`([^`]+)`', r'\1', text)

    # Links: [text](url) → text  (drop URLs — they don't need translation
    # and the English fallback already shows them)
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)

    # Heading markers: ## Heading → Heading
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)

    # Simplify bullet / list markers to a dash
    text = re.sub(r'^[\s]*[*+]\s+', '- ', text, flags=re.MULTILINE)

    # Horizontal rules
    text = re.sub(r'^[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)

    # Collapse runs of 3+ blank lines into two
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()


def _chunk_text(text: str, max_chars: int = _MAX_CHUNK_CHARS) -> list[str]:
    """Split *text* into chunks suitable for translation.

    Strategy (in order of preference):
      1. Split by paragraphs (double newline).
      2. If a paragraph is still too long, split by lines.
      3. If a single line is still too long, split by sentences.
    """
    paragraphs = re.split(r'\n\s*\n', text)
    chunks: list[str] = []

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # Fast path — paragraph fits in one chunk
        if len(para) <= max_chars:
            chunks.append(para)
            continue

        # Split the paragraph into lines and group them
        lines = para.split('\n')
        current = ""

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if len(line) > max_chars:
                # Flush what we have so far
                if current:
                    chunks.append(current)
                    current = ""
                # Split this oversized line by sentences
                sentences = re.split(r'(?<=[.!?])\s+', line)
                sent_buf = ""
                for sent in sentences:
                    if len(sent_buf) + len(sent) + 1 <= max_chars:
                        sent_buf += (" " if sent_buf else "") + sent
                    else:
                        if sent_buf:
                            chunks.append(sent_buf)
                        sent_buf = sent
                if sent_buf:
                    chunks.append(sent_buf)

            elif len(current) + len(line) + 1 <= max_chars:
                current += ("\n" if current else "") + line
            else:
                if current:
                    chunks.append(current)
                current = line

        if current:
            chunks.append(current)

    return chunks if chunks else [text[:max_chars]]


def translate_to_english(text: str) -> tuple[str, bool]:
    """Translate Romanian text to English.

    Returns a tuple of ``(translated_text, was_translated)``.
    If the input is already English, returns ``(text, False)``.
    """
    lang = detect_language(text)
    if lang == "eng":
        return text, False
    return _translate(text, src_lang="ron", tgt_lang="eng"), True


def translate_to_romanian(text: str) -> str:
    """Translate English text to Romanian.

    Strips markdown formatting and splits the text into manageable
    chunks before translating, to avoid token-limit truncation and
    keep the translation model focused on natural language.
    """
    clean = _strip_markdown(text)
    chunks = _chunk_text(clean)

    translated_parts: list[str] = []
    for chunk in chunks:
        translated_parts.append(
            _translate(chunk, src_lang="eng", tgt_lang="ron")
        )

    return "\n\n".join(translated_parts)
