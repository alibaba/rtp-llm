"""Bounded decoding of typed gRPC engine error metadata."""
import base64
import hashlib


def error_trailer_evidence(exc, pb2):
    """Record typed engine errors without interpreting transport cancellation.

    Missing, ambiguous or malformed metadata never becomes a zero error code.
    Keep at most 64 KiB of each of the first two matching values; retain length
    and SHA256 for oversized values. gRPC itself bounds received metadata.
    """
    result = dict(trailer_error_code=None, trailer_error_message=None)
    evidence = dict(status="absent", values=[])
    result["error_trailer"] = evidence
    try:
        read = getattr(exc, "trailing_metadata", None)
        metadata = read() if callable(read) else None
        values = [v for k, v in (metadata or ()) if k == "grpc-status-details-bin"]
        for value in values[:2]:
            if not isinstance(value, bytes):
                evidence["values"].append(dict(value_type=type(value).__name__))
                continue
            evidence["values"].append(
                dict(
                    base64=base64.b64encode(value[:65536]).decode("ascii"),
                    size_bytes=len(value),
                    sha256=hashlib.sha256(value).hexdigest(),
                    truncated=len(value) > 65536,
                )
            )
        evidence["value_count"] = len(values)
        if not values:
            return result
        if len(values) != 1:
            evidence["status"] = "ambiguous"
            return result
        raw = values[0]
        if not isinstance(raw, bytes) or not raw or len(raw) > 65536:
            evidence["status"] = "invalid_value"
            return result
        details = pb2.ErrorDetailsPB.FromString(raw)
        # Proto3 defaults alone are not evidence that a typed code was sent.
        # ListFields distinguishes an unknown-only/wrong-message payload.
        if not any(field.name == "error_code" for field, _ in details.ListFields()):
            evidence["status"] = "missing_error_code"
            return result
        result["trailer_error_code"] = int(details.error_code)
        result["trailer_error_message"] = details.error_message
        evidence["status"] = "parsed"
    except Exception as error:
        evidence.update(status="parse_error", error=repr(error))
    return result
