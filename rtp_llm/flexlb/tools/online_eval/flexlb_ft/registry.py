"""Explicit, ordered case registration shared by Python case modules."""

from .context import CaseDef
from .harness import PROFILE_CAPS, PROFILES


def case(
    name, *, category, profiles=None, requires=None, source="", expected_fail=False
):
    """Attach a case definition without mutating an import-time global list."""

    def decorate(fn):
        if hasattr(fn, "__case_def__"):
            raise ValueError(f"case already declared: {fn.__module__}.{fn.__name__}")
        fn.__case_def__ = CaseDef(
            name=name,
            category=category,
            fn=fn,
            profiles=profiles,
            requires=requires,
            source=source,
            expected_fail=expected_fail,
        )
        return fn

    return decorate


def validate_cases(definitions):
    """Reject ambiguous names and invalid metadata before runner filtering."""
    seen = {}
    capabilities = set().union(*PROFILE_CAPS.values())
    for definition in definitions:
        origin = f"{definition.fn.__module__}.{definition.fn.__name__}"
        if not isinstance(definition.name, str) or not definition.name.strip():
            raise ValueError(f"empty case name at {origin}")
        if definition.name in seen:
            raise ValueError(
                f"duplicate case {definition.name!r}: {seen[definition.name]} and {origin}"
            )
        seen[definition.name] = origin
        unknown_profiles = set(definition.profiles or ()) - set(PROFILES)
        unknown_requires = set(definition.requires or ()) - capabilities
        if unknown_profiles or unknown_requires:
            raise ValueError(
                f"invalid case metadata at {origin}: profiles={sorted(unknown_profiles)}, "
                f"requires={sorted(unknown_requires)}"
            )
    return definitions


def collect_cases(category, functions):
    """Collect exactly the explicit function order declared by a category."""
    definitions = []
    for fn in functions:
        definition = getattr(fn, "__case_def__", None)
        if definition is None or definition.fn is not fn:
            raise ValueError(f"missing @case definition: {fn}")
        if definition.category != category:
            raise ValueError(
                f"case {definition.name!r}: category {definition.category!r}, expected {category!r}"
            )
        definitions.append(definition)
    return validate_cases(definitions)
