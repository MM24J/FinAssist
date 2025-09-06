import re

ADVICE_HINTS = {
    "verbs": {"how", "what", "tips", "tricks", "reduce", "lower", "cut", "save", "optimize", "explain", "tell", "help"},
    "domains": {"internet", "phone", "mobile", "restaurant", "dining", "fast food", "coffee", "bill", "bills", 
                "emergency", "fund", "rule", "variance", "budgeting", "advice", "50/30/20", "savings"}
}

def looks_like_advice(ql: str) -> bool:
    # Check for advice verbs
    if any(v in ql for v in ADVICE_HINTS["verbs"]):
        return True
    
    # Check for financial advice domains
    if any(d in ql for d in ADVICE_HINTS["domains"]):
        return True
    
    # Special patterns for common advice questions
    advice_patterns = [
        "50/30/20",
        "emergency fund", 
        "variance",
        "rule",
        "should i",
        "how much",
        "how big"
    ]
    
    if any(pattern in ql for pattern in advice_patterns):
        return True
        
    return False

def looks_like_budget(ql: str) -> bool:
    # Explicit budget keywords
    budget_keywords = [
        "budget", "over budget", "under budget", "variance", "category", "spend",
        "over-budget", "under-budget", "spending", "categories"
    ]
    
    # Strong budget indicators
    if any(k in ql for k in budget_keywords):
        return True
    
    # Pattern matching for budget analysis requests
    if "show me my" in ql and ("top" in ql or "categories" in ql):
        return True
        
    if "top" in ql and "categories" in ql:
        return True
    
    # Date patterns often indicate budget analysis
    if re.search(r"\d{4}-\d{2}", ql):  # Matches "2019-09" format
        return True
        
    return False

def looks_like_invest(ql: str) -> bool:
    return any(k in ql for k in ["stock","ticker","price","return","market","invest","compare"])
