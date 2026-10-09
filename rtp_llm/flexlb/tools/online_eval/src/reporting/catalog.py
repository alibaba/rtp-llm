"""Generic report palette and UI colors.

Metric identities are data keys; names, axes and color are presentation only.
"""

PALETTE = (
    "#1677ff", "#52c41a", "#faad14", "#f5222d",
    "#722ed1", "#13c2c2", "#eb2f96", "#fa8c16",
    "#a0d911", "#2f54eb", "#fadb14", "#08979c",
)
ACCENT_COLORS = {
    "blue": "#2563eb", "amber": "#d97706", "green": "#059669",
    "red": "#be123c", "gray": "#e5e7eb",
}

TONE_TO_COLOR = {
    "primary": PALETTE[0], "success": PALETTE[1],
    "warning": PALETTE[2], "warn": PALETTE[2],
    "danger": PALETTE[3], "info": PALETTE[5],
    "secondary": PALETTE[4], "tertiary": PALETTE[6],
    "quaternary": PALETTE[7], "neutral": "#8c8c8c",
}
KPI_TONE_COLOR = {
    "success": PALETTE[1], "danger": PALETTE[3],
    "warn": PALETTE[2], "warning": PALETTE[2],
    "info": PALETTE[0], "primary": PALETTE[0],
}


def series_color(tone, index):
    return TONE_TO_COLOR.get(tone, PALETTE[index % len(PALETTE)])


RENDERER_THEME = {
    "PRIMARY": "#1677ff",
    "DARK_TEXT": "#333",
    "SUCCESS": "#52c41a",
    "EVENT_TEXT": "#64748b",
    "MUTED_TEXT": "#888",
    "EVENT_LINE": "#94a3b8",
    "HOVER_BORDER": "#999",
    "BORDER": "#d9d9d9",
    "LIGHT_BORDER": "#ddd",
    "HOVER_BACKGROUND": "#eef4ff",
    "DANGER": "#f5222d",
    "BACKGROUND": "#f5f6fa",
    "CODE_BACKGROUND": "#f7f8fa",
    "WARNING": "#faad14",
    "MUTED_BACKGROUND": "#fafafa",
    "CARD": "#fff",
}
