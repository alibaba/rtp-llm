"""Metric identity syntax, independent of origin and presentation names."""

import re

NAME = re.compile(r"[a-z][a-z0-9_]*\Z")
METRIC_ID = re.compile(r"[a-z][a-z0-9_]*/[a-z][a-z0-9_]*\Z")
