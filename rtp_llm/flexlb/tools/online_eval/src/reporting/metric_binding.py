"""Local view curves select canonical metric IDs and explicit label projections."""


def bindings(presentation, metric_id, labels):
    return [(curve_id, style) for curve_id, style in presentation["curves"].items()
            if style["metric_id"] == metric_id
            and all(labels.get(key) == value for key, value in style["labels"].items())]
