import os, re, pickle
from typing import List, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer

# Paths
KB_PATH = os.path.join(os.path.dirname(__file__), "..", "kb", "finance_guide.md")
EMB_PATH = os.path.join(os.path.dirname(__file__), "..", "kb", "embeddings.pkl")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Special case: 50/30/20 rule definition
FIFTY_RULE_LINE = (
    "Allocate ~50% to needs, 30% to wants, 20% to savings/debt "
    "(e.g., on $3,000 take-home: $1,500 needs / $900 wants / $600 savings)."
)

# Domain-specific keywords boosts
SECTION_KEYWORDS = {
    "internet": {"internet", "provider", "promo", "bundle", "auto-pay", "autopay"},
    "mobile": {"mobile", "phone", "provider", "plan", "promo", "bundle", "auto-pay", "autopay"},
    "restaurnat": {"restaurant", "dining", "fast", "coffee", "cap", "track", "meal", "prep"},
    "emergency": {"emergency", "fund", "safety", "savings"}
}

# Embedding + Index

def _load_chunks(path: str, max_len = 800) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    
    # repack long blocks into <= max_len
    chunks, buf = [], ""
    for b in blocks:
        if not buf:
            buf = b
        elif len(buf) + 1 + len(b) <= max_len:
            but = f"{buf}\n{b}"
        else:
            chunks.append(buf)
            buf = b
    if buf:
        chunks.append(buf)
    return chunks

def _embed_chunks(chunks: List[str]):
    model = SentenceTransformer(MODEL_NAME)
    return model.encode(chunks, normalize_embeddings=True, show_progress_bar=False)

def _ensure_index():
    os.makedirs(os.path.dirname(EMB_PATH), exist_ok=True)
    if os.path.exists(EMB_PATH):
        with open(EMB_PATH, "rb") as f:
            data = pickle.load(f)
        if data.get("model") == MODEL_NAME and os.path.exists(KB_PATH):
            return data["chunks"], data["emb"]
    chunks = _load_chunks(KB_PATH)
    emb = _embed_chunks(chunks)
    with open(EMB_PATH, "wb") as f:
        pickle.dump({"model": MODEL_NAME, "chunks": chunks, "emb": emb}, f)
    return chunks, emb

# Search
def _search(query: str, k=3) -> List[Tuple[str, float]]:
    chunks, emb = _ensure_index()
    model = SentenceTransformer(MODEL_NAME)
    qv = model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
    sims = emb @ qv
    idx = np.argsort(-sims)[:k*3]
    
    qwords = set(re.findall(r"[a-z]{3,}", query.lower()))
    key_hits = []
    for i in idx:
        text = chunks[i].lower()
        if any(w in text for w in qwords):
            key_hits.append((chunks[i], float(sims[i])))
            
    if not key_hits:
        key_hits = [(chunks[i], float(sims[i])) for i in idx]
    
    return key_hits[:k]

# Answer construction
def _extract_bullets(context: str):
    lines = [ln.strip() for ln in context.splitlines()]
    return [ln.lstrip("-• ").strip() for ln in lines if ln and (ln.startswith(("-", ".")) or len(ln) <= 120)]

def _keywordize(text: str):
    return set(re.findall(r"[a-zA-Z]{3,}", text.lower()))

def _q_terms(q: str):
    ql = q.lower()
    terms = _keywordize(ql)
    boost = set()
    for key, kws in SECTION_KEYWORDS.items():
        if key in ql:
            boost |= kws
    return terms, boost

def _score_line(line: str, qterms: set, boost: set):
    lw = set(re.findall(r"[a-zA-Z]{3,}", line.lower()))
    base = len(lw & qterms)
    b = 1.0 if (lw & boost) else 0.0
    length_bonus = 0.2 if len(line) <= 100 else 0.0
    return base + b + length_bonus

def _dollar_examples(line: str, q: str):
    ql = q.lower()
    if any(k in ql for k in ["restaurant", "dining", "fast food", "coffeeshop"]):
        if "cap" in line or "track" in line:
            return line + " (e.g., set a weekly dining cap of $40-60 and track spend)."
        if "meal prep" in line:
            return line + " (e.g., two lunches/week could save ~$20-$30)."
    if "internet" in ql or "mobile" in ql or "phone" in ql:
        if "promo" in line or "provider" in line:
            return line + " (e.g., ask for promo pricing; $5-$20/mo. savings is common)."
        if "auto-pay" in line or "bundle" in line:
            return line + " (e.g., bundles/auto-pay often save 5-15%, e.g., $5-$15)."
    return line

def _filter_bullets_by_keywords(bullets: list[str], q_terms: set, boost: set):
    """
    Keep bullets that overlap with boost terms (like 'internet', 'restaurant')
    OR with the general query terms.
    But if that leaves us with fewer than 2 bullets, fall back to all bullets.
    """
    if not boost:
        return bullets

    filtered = []
    for ln in bullets:
        lw = set(re.findall(r"[a-zA-Z]{3,}", ln.lower()))
        if (lw & boost) or (lw & q_terms):
            filtered.append(ln)

    # Don’t over-filter: keep at least 2 bullets
    if len(filtered) < 2:
        return bullets  

    return filtered

def _make_answer(question: str, context: str, k_keep=5):
    q_terms, boost = _q_terms(question)
    bullets = _extract_bullets(context)
    print("DEBUG bullets:", bullets)   # 👈 Debug

    if not bullets:
        para = context.split("\n\n")[0].strip()
        return f"**Answer:** {question.strip()}\n{para[:600]}...\n\n_Source: FinAssist KB_"

    filtered = _filter_bullets_by_keywords(bullets, q_terms, boost)
    print("DEBUG filtered:", filtered)  # 👈 Debug

    if not filtered:
        filtered = bullets  

    # Score and rank
    scored = sorted(
        ((ln, _score_line(ln, q_terms, boost)) for ln in filtered),
        key=lambda t: t[1],
        reverse=True,
    )

    # Deduplicate & keep top N
    seen, picked = set(), []
    for ln, _ in scored:
        key = ln.lower()
        if key in seen:
            continue
        seen.add(key)
        picked.append(_dollar_examples(ln, question))
        if len(picked) >= k_keep:
            break

    if not picked:  # absolute fallback
        return f"**Answer:** {question.strip()}\n(context found but no clean bullets)\n\n_Source: FinAssist KB_"

    # Format final output
    out = [f"**Answer:** {question.strip()}"]
    out += [f"• {p}" for p in picked]
    out.append("\n_Source: FinAssist KB_")
    return "\n".join(out)



# Public API
def rag_answer(question: str, k: int=3) -> str:
    if "50/30/20" in question.lower():
        return ("**Answer:** What is the 50/30/20 rule?\n"
                f"• {FIFTY_RULE_LINE}\n"
                "\n_Source: FinAssist KB_")
        
    hits = _search(question, k=k)
    if not hits:
        return "I don't have enough info to answer that."
    context = "\n\n---\n\n".join([c for c, _ in hits])
    return _make_answer(question, context, k_keep=5)