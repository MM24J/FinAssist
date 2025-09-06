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
    "restaurant": {"restaurant", "dining", "fast", "coffee", "cap", "track", "meal", "prep"},  # Fixed typo: "restaurnat" -> "restaurant"
    "emergency": {"emergency", "fund", "safety", "savings"}
}

# Embedding + Index

def _load_chunks(path: str, max_len = 800) -> List[str]:
    """Load and chunk the finance guide markdown file."""
    print(f"DEBUG: Loading chunks from {path}")
    
    if not os.path.exists(path):
        print(f"ERROR: File does not exist: {path}")
        return []
    
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    
    # repack long blocks into <= max_len
    chunks, buf = [], ""
    for b in blocks:
        if not buf:
            buf = b
        elif len(buf) + 1 + len(b) <= max_len:
            buf = f"{buf}\n{b}"  # FIXED: Was "but = f"{buf}\n{b}"
        else:
            chunks.append(buf)
            buf = b
    if buf:
        chunks.append(buf)
    
    return chunks

def _embed_chunks(chunks: List[str]):
    print(f"DEBUG: Embedding {len(chunks)} chunks")
    model = SentenceTransformer(MODEL_NAME)
    return model.encode(chunks, normalize_embeddings=True, show_progress_bar=False)

def _ensure_index():
    os.makedirs(os.path.dirname(EMB_PATH), exist_ok=True)
    
    # Force rebuild if we detect the old small file
    if os.path.exists(EMB_PATH):
        with open(EMB_PATH, "rb") as f:
            data = pickle.load(f)
        if data.get("model") == MODEL_NAME and os.path.exists(KB_PATH):
            # Check if we have a valid knowledge base
            if len(data['chunks']) < 5 or sum(len(chunk) for chunk in data['chunks']) < 1000:
                print("DEBUG: Detected small/invalid cached embeddings, forcing rebuild...")
                os.remove(EMB_PATH)
            else:
                print(f"DEBUG: Using cached embeddings with {len(data['chunks'])} chunks")
                return data["chunks"], data["emb"]
    
    print("DEBUG: Creating new embeddings...")
    chunks = _load_chunks(KB_PATH)
    if not chunks:
        print("ERROR: No chunks loaded! Check your finance_guide.md file")
        return [], np.array([])
    
    emb = _embed_chunks(chunks)
    with open(EMB_PATH, "wb") as f:
        pickle.dump({"model": MODEL_NAME, "chunks": chunks, "emb": emb}, f)
    return chunks, emb

# Search
def _search(query: str, k=3) -> List[Tuple[str, float]]:
    print(f"DEBUG: Searching for: '{query}'")
    chunks, emb = _ensure_index()
    
    if len(chunks) == 0:
        print("ERROR: No chunks available for search!")
        return []
    
    model = SentenceTransformer(MODEL_NAME)
    qv = model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
    sims = emb @ qv
    idx = np.argsort(-sims)[:k*3]
    
    print(f"DEBUG: Top similarities: {sims[idx[:3]]}")
    
    qwords = set(re.findall(r"[a-z]{3,}", query.lower()))
    print(f"DEBUG: Query words: {qwords}")
    
    key_hits = []
    for i in idx:
        text = chunks[i].lower()
        if any(w in text for w in qwords):
            key_hits.append((chunks[i], float(sims[i])))
            
    if not key_hits:
        key_hits = [(chunks[i], float(sims[i])) for i in idx]
    
    print(f"DEBUG: Found {len(key_hits)} relevant chunks")
    return key_hits[:k]

# Answer construction
def _extract_bullets(context: str):
    """Extract bullet points from context."""
    bullets = []
    
    # First, try to extract standard bullets
    for ln in context.splitlines():
        line = ln.strip()
        # Handle various bullet formats: -, •, *, and also lines that start with "Allocate" etc.
        if line.startswith(("-", "•", "*")) or (len(line) > 10 and any(line.startswith(word) for word in ["Allocate", "Set", "Meal", "Move", "Call", "Bundle", "Target", "Automate", "Treat", "Use", "Compare"])):
            clean_line = line.lstrip("-•* ").strip()
            if clean_line:  # Only add non-empty lines
                bullets.append(clean_line)
    
    # If no bullets found, try to extract sentences or meaningful lines
    if not bullets:
        # Look for the 50/30/20 rule specifically
        if "50/30/20" in context or "50%" in context:
            lines = context.split('\n')
            for line in lines:
                line = line.strip()
                # Extract lines that contain percentages or allocation info
                if any(word in line.lower() for word in ["50%", "30%", "20%", "allocate", "needs", "wants", "savings"]):
                    if len(line) > 15:  # Meaningful content
                        bullets.append(line)
        
        # General fallback: extract any substantial lines
        if not bullets:
            lines = [line.strip() for line in context.split('\n') if line.strip()]
            bullets = [line for line in lines if len(line) > 20 and ('.' in line or ',' in line)]
    
    return bullets

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
    Keep bullets that overlap with boost terms OR with the general query terms.
    Use stricter filtering for better relevance.
    """
    # Special case: for 50/30/20 rule questions, prioritize allocation bullets
    if any("50" in term or "rule" in term for term in q_terms):
        allocation_bullets = []
        other_bullets = []
        
        for ln in bullets:
            if any(word in ln.lower() for word in ["allocate", "50%", "30%", "20%", "needs", "wants", "savings"]):
                allocation_bullets.append(ln)
            else:
                other_bullets.append(ln)
        
        if allocation_bullets:
            return allocation_bullets[:3]  # Return just the allocation bullets
    
    if not boost and not q_terms:
        return bullets

    # First pass: strong matches with boost keywords
    strong_matches = []
    weak_matches = []
    
    for ln in bullets:
        lw = set(re.findall(r"[a-zA-Z]{3,}", ln.lower()))
        
        # Strong match: overlaps with boost keywords (domain-specific)
        if boost and (lw & boost):
            strong_matches.append(ln)
        # Weak match: overlaps with general query terms
        elif lw & q_terms:
            weak_matches.append(ln)
    
    # Prefer strong matches, fall back to weak matches if needed
    if strong_matches:
        return strong_matches[:5]  # Limit to top 5 most relevant
    elif weak_matches:
        return weak_matches[:3]   # Fewer weak matches
    else:
        return bullets[:2]        # Minimal fallback

def _make_answer(question: str, context: str, k_keep=5):
    print(f"DEBUG: Making answer for: '{question}'")
    print(f"DEBUG: Context length: {len(context)} chars")
    
    q_terms, boost = _q_terms(question)
    bullets = _extract_bullets(context)
    print("DEBUG all bullets:", bullets[:10])

    if not bullets:
        # Try to extract any lines that look like advice
        lines = [line.strip() for line in context.split('\n') if line.strip()]
        bullets = [line for line in lines if len(line) > 20 and ('.' in line or ',' in line)]
        print(f"DEBUG fallback bullets from lines: {bullets[:5]}")
        
        if not bullets:
            para = context.split("\n\n")[0].strip()
            result = f"**Answer:** {question.strip()}\n{para[:600]}...\n\n_Source: FinAssist KB_"
            print(f"DEBUG: Returning fallback result: {result}")
            return result

    filtered = _filter_bullets_by_keywords(bullets, q_terms, boost)
    print("DEBUG filtered:", filtered)

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
    for ln, score in scored:
        key = ln.lower()
        if key in seen:
            continue
        seen.add(key)
        picked.append(_dollar_examples(ln, question))
        if len(picked) >= k_keep:
            break

    print(f"DEBUG: Final picked bullets: {picked}")

    if not picked:  # absolute fallback
        result = f"**Answer:** {question.strip()}\n(context found but no clean bullets)\n\n_Source: FinAssist KB_"
        print(f"DEBUG: Returning no-bullets fallback: {result}")
        return result

    # Format final output
    out = [f"**Answer:** {question.strip()}"]
    out += [f"• {p}" for p in picked]
    out.append("\n_Source: FinAssist KB_")
    
    final_result = "\n".join(out)
    print(f"DEBUG: Final formatted result:\n{final_result}")
    return final_result

def rag_answer(question: str, k: int = 3) -> str:
    """
    Main RAG function that takes a question and returns an answer.
    This is the function that should be imported by other modules.
    
    Args:
        question: The user's question
        k: Number of chunks to retrieve (default: 3)
    """
    # Add debug for 50/30/20 questions
    if "50/30/20" in question or "rule" in question.lower():
        print(f"DEBUG RAG_ANSWER: Processing question '{question}'")
    
    # Search for relevant context
    search_results = _search(question, k=k)
    
    if not search_results:
        return f"**Answer:** {question.strip()}\nI couldn't find relevant information in the knowledge base. Please check if your finance guide is properly loaded.\n\n_Source: FinAssist KB_"
    
    # Debug the search results for rule questions
    if "50/30/20" in question or "rule" in question.lower():
        print(f"DEBUG: Got {len(search_results)} search results")
        for i, (chunk, score) in enumerate(search_results):
            print(f"DEBUG: Result {i+1} (score: {score:.3f}):\n{chunk[:200]}...\n")
    
    # Combine all search results into context
    context = "\n\n".join([chunk for chunk, score in search_results])
    
    if "50/30/20" in question or "rule" in question.lower():
        print(f"DEBUG: Combined context length: {len(context)}")
        print(f"DEBUG: Context content:\n{context}\n")
    
    # Generate answer
    answer = _make_answer(question, context)
    
    if "50/30/20" in question or "rule" in question.lower():
        print(f"DEBUG: Final answer:\n{answer}\n")
    
    return answer

# Add a test function to help debug
def debug_kb_loading():
    """Test function to debug knowledge base loading."""
    print("=== DEBUG KB LOADING ===")
    chunks, emb = _ensure_index()
    print(f"Loaded {len(chunks)} chunks")
    
    if chunks:
        print(f"First chunk preview: {chunks[0][:300]}...")
        bullets = _extract_bullets(chunks[0])
        print(f"Bullets in first chunk: {bullets}")
    
    # Test search
    test_query = "How can I save money on restaurants?"
    results = _search(test_query)
    print(f"\nTest search results for '{test_query}':")
    for i, (chunk, score) in enumerate(results):
        print(f"  Result {i+1} (score: {score:.3f}): {chunk[:100]}...")
    
    # Test the full rag_answer function
    print(f"\n=== TESTING RAG_ANSWER ===")
    answer = rag_answer(test_query)
    print(f"Final answer: {answer}")

if __name__ == "__main__":
    debug_kb_loading()