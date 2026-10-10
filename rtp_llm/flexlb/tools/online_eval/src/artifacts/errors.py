"""Artifact absence is distinct from malformed or corrupted committed data."""


class ArtifactNotProduced(FileNotFoundError):
    pass
