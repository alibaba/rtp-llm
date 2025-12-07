"""
Utility functions for response processing and aggregation
"""

from typing import List, Optional, TypeVar

T = TypeVar("T")


def append_string_field(
    existing_value: Optional[str], delta_value: Optional[str]
) -> Optional[str]:
    """
    Concatenate string fields, handling None cases

    Args:
        existing_value: Existing string value (may be None)
        delta_value: New string value to append (may be None)

    Returns:
        Concatenated string or None
    """
    if existing_value is None:
        return delta_value or None
    return existing_value + (delta_value or "")


def update_field_with_latest(
    existing_value: Optional[T], new_value: Optional[T]
) -> Optional[T]:
    """
    Update field with latest non-None value

    Args:
        existing_value: Existing field value
        new_value: New field value

    Returns:
        new_value if not None, otherwise existing_value
    """
    return new_value or existing_value


def initialize_if_none(value: Optional[T], default_factory) -> T:
    """
    Initialize value if None using default factory

    Args:
        value: Value to check
        default_factory: Factory function to create default value

    Returns:
        value if not None, otherwise result of default_factory()
    """
    return value if value is not None else default_factory()
