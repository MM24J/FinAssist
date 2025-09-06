ADVICE_HINTS = {
    "verbs": {"how", "tips", "tricks", "reduce", "lower", "cut", "save", "optimize", "explain"},
    "domains": {"internet", "phone", "mobile", "restaurant", "dining", "fast food", "coffee", "bill", "bills"}
}

def looks_like_advice(ql: str) -> bool:
    if any(v in ql for v in ADVICE_HINTS["verbs"]):
        return True
    if any(d in ql for d in ADVICE_HINTS["domains"]):
        return True
    return False

def looks_like_budget(ql: str) -> bool:
    return any(k in ql for k in ["budget","over budget","under budget","variance","category","spend"])

def looks_like_invest(ql: str) -> bool:
    return any(k in ql for k in ["stock","ticker","price","return","market","invest","compare"])
