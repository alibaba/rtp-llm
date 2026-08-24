import re
from typing import Any, Dict, List


def validate_semantic_response(response: Any, config: Dict[str, Any]) -> List[str]:
    """Return human-readable semantic validation failures for a smoke response."""
    if not isinstance(response, str):
        return [f"semantic response must be a string, got {type(response).__name__}"]

    text = response.strip()
    failures: List[str] = []
    semantic_text = text
    terminal_sequences = [
        str(sequence)
        for sequence in config.get("terminal_sequences", [])
        if str(sequence)
    ]
    if terminal_sequences:
        matched_terminal = False
        for sequence in terminal_sequences:
            if sequence not in semantic_text:
                continue
            matched_terminal = True
            semantic_text, trailing = semantic_text.split(sequence, 1)
            if trailing.strip():
                failures.append(
                    "semantic response contains text after terminal sequence "
                    f"{sequence!r}: {trailing[:120]!r}"
                )
            break
        if not matched_terminal:
            failures.append(
                "semantic response is missing required terminal sequence: "
                + " | ".join(repr(sequence) for sequence in terminal_sequences)
            )
    else:
        for marker in config.get("stop_markers", ["[EOS]"]):
            if marker not in semantic_text:
                continue
            semantic_text, trailing = semantic_text.split(marker, 1)
            if trailing.strip():
                failures.append(
                    f"semantic response contains text after stop marker {marker!r}: {trailing[:120]!r}"
                )
            break

    semantic_text = semantic_text.strip()
    minimum_chars = int(config.get("minimum_chars", 20))
    if len(semantic_text) < minimum_chars:
        failures.append(
            f"semantic response is too short: {len(semantic_text)} chars, "
            f"expected at least {minimum_chars}"
        )

    lowered = semantic_text.casefold()
    for alternatives in config.get("required_concept_groups", []):
        if not any(str(value).casefold() in lowered for value in alternatives):
            failures.append(
                "semantic response is missing required concept group: "
                + " | ".join(map(str, alternatives))
            )

    if config.get("reject_repetition", True):
        segments = []
        for value in re.split(r"[\n。！？!?；;]+", semantic_text):
            normalized = re.sub(r"[\s*_`#>\-]+", "", value).casefold()
            if len(normalized) >= 8:
                segments.append(normalized)
        seen = set()
        for segment in segments:
            if segment in seen:
                failures.append(
                    f"semantic response repeats a sentence or paragraph: {segment[:80]!r}"
                )
                break
            seen.add(segment)

        compact = re.sub(r"\s+", "", semantic_text).casefold()
        max_span = min(96, len(compact) // 2)
        for span in range(max_span, 11, -1):
            duplicate = next(
                (
                    compact[start : start + span]
                    for start in range(len(compact) - 2 * span + 1)
                    if compact[start : start + span]
                    == compact[start + span : start + 2 * span]
                ),
                None,
            )
            if duplicate is not None:
                failures.append(
                    f"semantic response contains an adjacent repeated span: {duplicate[:80]!r}"
                )
                break

    return failures
