"""Recognize one explicitly named format; never infer it from a numeric version."""


def version_fields(document):
    return {key for key in document
            if isinstance(key, str) and (key == "schema_version" or key.endswith("_schema_version"))}


def matches_schema(document, field, version):
    """Reject missing, foreign, mixed and non-integer version headers."""
    return (isinstance(document, dict)
            and version_fields(document) == {field}
            and type(document[field]) is int and document[field] == version)
