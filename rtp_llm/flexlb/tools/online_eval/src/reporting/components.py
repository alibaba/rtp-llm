"""Presentation-only chart and KPI components."""


def chart_spec(chart_type, categories, series, suffix=None, y_max=None):
    """直接保留列表数据；颜色顺序与渲染器一致。"""
    from reporting.renderer import series_color

    return {
        "type": chart_type,
        "x": categories,
        "yMax": y_max,
        "unit": suffix.strip() if suffix else "",
        "series": [
            {
                "name": name,
                "data": data,
                "color": series_color(tone, index),
                "tone": tone or "",
            }
            for index, (_key, name, data, tone) in enumerate(series)
        ],
    }


def panel_spec(title, caption, chart):
    return dict(chart, title=title, caption=caption)


def kpi_spec(value, label, tone=None):
    return {
        "label": label,
        "value": str(value),
        "tone": {"warning": "warn"}.get(tone, tone or ""),
    }
