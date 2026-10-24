"""Indexed lossless containers; independent of analysis and presentation."""

from __future__ import annotations

import hashlib
import json
import tempfile
import zipfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path, PurePosixPath

from artifacts.archive import PRECOMPRESSED
from schema_contract import matches_schema


class ArtifactRole(str, Enum):
    DEFINITION = "definition"
    EXECUTION = "execution"
    OBSERVATIONS = "observations"
    GATE_INPUT = "gate_input"
    GATE_RESULT = "gate_result"
    REQUESTS = "requests"
    CONFIGURATION = "configuration"
    VIEW = "view"
    ANALYSIS = "analysis"
    EXTENSION = "extension"
    RAW = "raw"


@dataclass(frozen=True)
class ArtifactEntry:
    path: str
    blob: str
    sha256: str
    size: int
    role: ArtifactRole
    origin: str | None = None

    def to_dict(self):
        return dict(asdict(self), role=self.role.value)

    @classmethod
    def read(cls, value):
        if not isinstance(value, dict) or set(value) != {"path", "blob", "sha256", "size", "role", "origin"}:
            raise ValueError("invalid experiment entry")
        path, digest = value["path"], value["sha256"]
        if (
            type(path) is not str
            or not path
            or PurePosixPath(path).is_absolute()
            or "\\" in path
            or any(part in {"", ".", ".."} for part in path.split("/"))
            or type(digest) is not str
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
            or value["blob"] != "blobs/" + digest
            or type(value["size"]) is not int
            or value["size"] < 0
            or value["origin"] is not None
            and type(value["origin"]) is not str
        ):
            raise ValueError("invalid experiment entry")
        return cls(path, value["blob"], digest, value["size"], ArtifactRole(value["role"]), value["origin"])


@dataclass(frozen=True)
class ViewDescriptor:
    layout: str
    declaration: dict | None

    @classmethod
    def read(cls, value):
        if (
            not isinstance(value, dict)
            or set(value) != {"layout", "declaration"}
            or type(value["layout"]) is not str
            or value["declaration"] is not None
            and not isinstance(value["declaration"], dict)
        ):
            raise ValueError("invalid experiment view")
        return cls(value["layout"], value["declaration"])


def _verify_gate_inputs(entries, read_json):
    for name in entries:
        if not name.endswith("-gate-manifest.json"):
            continue
        document = read_json(name)
        if not matches_schema(document, "gate_result_schema_version", 1):
            raise ValueError("invalid frozen gate manifest")
        stem = name[: -len("-manifest.json")]
        for key in ("evidence", "result"):
            target = stem + "-" + key + ".json"
            reference = document[key]
            if (
                reference["file"] != PurePosixPath(target).name
                or target not in entries
                or entries[target].sha256 != reference["sha256"]
            ):
                raise ValueError("frozen gate evidence checksum mismatch")


class ExperimentWriter:
    """Atomic ZIP publication, one content-addressed blob per distinct payload."""

    def __init__(self, output, kind="experiment", deadline=None):
        self.output = Path(output)
        self.kind = kind
        self.deadline = deadline
        self.entries = {}
        self.blobs = set()
        self.views = {}
        self.omitted = []

    def check(self):
        if self.deadline is not None:
            self.deadline.check()

    @contextmanager
    def publication(self):
        self.output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=self.output.parent, prefix=".experiment-") as directory:
            temporary = Path(directory) / "artifact.zip"
            self.scratch = Path(directory)
            with zipfile.ZipFile(
                temporary, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=3, allowZip64=True
            ) as archive:
                self.archive = archive
                yield self
                self.check()
                _verify_gate_inputs(self.entries, lambda name: json.loads(archive.read(self.entries[name].blob)))
                manifest = dict(
                    experiment_schema_version=1,
                    kind=self.kind,
                    files={name: entry.to_dict() for name, entry in self.entries.items()},
                    views={name: asdict(view) for name, view in self.views.items()},
                    omitted=self.omitted,
                )
                archive.writestr(
                    zipfile.ZipInfo("manifest.json"),
                    json.dumps(manifest, ensure_ascii=False, allow_nan=False, separators=(",", ":")),
                    compress_type=zipfile.ZIP_DEFLATED,
                    compresslevel=3,
                )
            self.check()
            temporary.replace(self.output)

    def add_file(self, name, path, role):
        self.check()
        path = Path(path)
        digest = hashlib.sha256()
        size = 0
        # Snapshot and hash in one pass: the committed bytes must match the digest.
        with tempfile.TemporaryFile(dir=self.scratch) as snapshot:
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    self.check()
                    digest.update(block)
                    size += len(block)
                    snapshot.write(block)
            sha = digest.hexdigest()
            blob = "blobs/" + sha
            entry = ArtifactEntry(name, blob, sha, size, role, str(path.resolve()))
            ArtifactEntry.read(entry.to_dict())
            if name in self.entries:
                raise ValueError("duplicate experiment path: " + name)
            if blob not in self.blobs:
                member = blob
                if path.suffix.lower() in PRECOMPRESSED:
                    member = zipfile.ZipInfo(blob)
                    member.compress_type = zipfile.ZIP_STORED
                snapshot.seek(0)
                with self.archive.open(member, "w", force_zip64=True) as target:
                    for block in iter(lambda: snapshot.read(1024 * 1024), b""):
                        self.check()
                        target.write(block)
                self.blobs.add(blob)
            self.entries[name] = entry
        return name

    def add_json(self, name, value, role):
        path = self.scratch / "document.json"
        with path.open("w", encoding="utf-8") as stream:
            stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")))
        self.add_file(name, path, role)
        # Generated payloads do not reference the temporary encoder directory.
        entry = self.entries[name]
        self.entries[name] = ArtifactEntry(entry.path, entry.blob, entry.sha256, entry.size, entry.role)
        return name


class ExperimentReader:
    def __init__(self, path):
        self.archive = zipfile.ZipFile(path)
        try:
            names = self.archive.namelist()
            if len(names) != len(set(names)):
                raise ValueError("duplicate experiment ZIP members")
            manifest = json.loads(self.archive.read("manifest.json"))
            if (
                not matches_schema(manifest, "experiment_schema_version", 1)
                or manifest.get("kind") not in {"experiment", "raw"}
                or not isinstance(manifest.get("files"), dict)
                or not isinstance(manifest.get("views"), dict)
            ):
                raise ValueError("invalid experiment manifest")
            self.kind = manifest["kind"]
            self.entries = {name: ArtifactEntry.read(value) for name, value in manifest["files"].items()}
            self.views = {name: ViewDescriptor.read(value) for name, value in manifest["views"].items()}
            self.omitted = manifest["omitted"]
            if any(name != entry.path for name, entry in self.entries.items()):
                raise ValueError("experiment path mismatch")
            expected = {"manifest.json"} | {entry.blob for entry in self.entries.values()}
            if set(names) != expected:
                raise ValueError("experiment ZIP inventory mismatch")
            if (
                self.kind == "experiment"
                and not {"experiment.json", "execution.json", "completeness.json"} <= self.entries.keys()
            ):
                raise ValueError("experiment lacks required facts")
            if any(
                view.layout not in self.entries or self.entries[view.layout].role is not ArtifactRole.VIEW
                for view in self.views.values()
            ):
                raise ValueError("experiment view layout is not indexed")
        except Exception:
            self.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self.archive.close()

    @contextmanager
    def open(self, name):
        if name == "run/result.json" and name not in self.entries:
            name = "execution.json"  # Read the historical alias without storing another entry.
        entry = self.entries[name]
        digest, size = hashlib.sha256(), 0
        # Verify while copying to a disk spool, rather than holding all bytes twice.
        with tempfile.TemporaryFile() as snapshot:
            with self.archive.open(entry.blob) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
                    size += len(block)
                    snapshot.write(block)
            if digest.hexdigest() != entry.sha256 or size != entry.size:
                raise ValueError("experiment checksum mismatch: " + name)
            snapshot.seek(0)
            yield snapshot

    def read_json(self, name):
        with self.open(name) as stream:
            return json.load(stream)

    def resolve_source(self, path):
        """Resolve frozen physical provenance without reading the original host."""
        if path in self.entries:
            return path
        matches = [name for name, entry in self.entries.items() if entry.origin == str(path)]
        if not matches:
            raise KeyError("source is not in this experiment: " + str(path))
        return matches[0]

    def verify(self):
        checked = set()
        for name, entry in self.entries.items():
            if entry.blob not in checked:
                with self.open(name):
                    pass
                checked.add(entry.blob)
        _verify_gate_inputs(self.entries, self.read_json)

    def verify_raw(self, path):
        from file_digest import sha256_file

        reference = self.read_json("raw-attachment.json")
        if Path(path).stat().st_size != reference["size"] or sha256_file(path) != reference["sha256"]:
            raise ValueError("raw attachment checksum mismatch")
        with ExperimentReader(path) as raw:
            if raw.kind != "raw":
                raise ValueError("invalid raw attachment kind")
            raw.verify()
