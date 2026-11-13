"""Validate the Grafana dashboard saved with a Mock run."""


def validate_dashboard(value):
    if (
        not isinstance(value, dict)
        or not isinstance(value.get("panels"), list)
        or not value.get("title")
    ):
        raise ValueError(
            "view must be a native Grafana dashboard JSON object with title and panels"
        )
