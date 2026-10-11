"""Render declared section labels and open state without interpreting results."""

from reporting.core import details, table
from reporting.view_model import SectionPresentation


def view_details(view, key, value):
    section = SectionPresentation.read(view["sections"][key])
    if section.columns is not None:
        raise ValueError("detail section cannot declare columns: " + key)
    return details(section.title, value, opened=section.opened, identity="case." + key)


def view_table(view, key, rows):
    section = SectionPresentation.read(view["sections"][key])
    columns = section.columns
    if columns is None:
        raise ValueError("table section must declare columns: " + key)
    rows = list(rows)
    if any(len(row) != len(columns) for row in rows):
        raise ValueError("section row does not match declared columns: " + key)
    return table(section.title, list(columns), rows, opened=section.opened, identity="case." + key)
