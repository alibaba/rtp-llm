def normalize_think_tag(value: str) -> str:
    """Convert the literal newline escapes accepted by THINK_*_TAG."""
    return value.replace(r"\n", "\n")
