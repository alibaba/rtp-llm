"""Render declared section labels and open state without interpreting results."""

from reporting.core import details, table


def view_details(view, key, value):
    section = view["sections"][key]
    if "columns" in section:
        raise ValueError("detail section cannot declare columns: " + key)
    return details(section["title"], value, opened=section["opened"])


def view_table(view, key, rows):
    section = view["sections"][key]
    columns = section["columns"]
    rows = list(rows)
    if any(len(row) != len(columns) for row in rows):
        raise ValueError("section row does not match declared columns: " + key)
    return table(section["title"], columns, rows, opened=section["opened"])
