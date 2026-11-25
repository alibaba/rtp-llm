"""Opt-in for the complete native remote EMB path."""

import os

USE_NATIVE_EMB = os.environ.get("MM_EMB_NATIVE") == "1"
