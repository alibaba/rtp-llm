"""Private SM120 sparse MLA implementation."""

# Larger query batches use the native prefill dispatcher.
DECODE_MAX_TOKENS = 64
