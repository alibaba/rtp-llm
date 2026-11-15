"""Publish this Bazel invocation's declared outputs for CI report collection."""

import argparse
import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from urllib.parse import unquote, urlsplit


def read_results(events, source, exit_code):
    results, files = [], {}
    finished = False
    with Path(events).open() as stream:
        for line in stream:
            event = json.loads(line)
            if "finished" in event:
                finished = (
                    event["finished"].get("exitCode", {}).get("code", 0) == exit_code
                )
            identity = event.get("id", {}).get("testResult")
            if identity is None or "testResult" not in event:
                continue
            result = event["testResult"]
            results.append(dict(identity, status=result.get("status")))
            has_xml = False
            for output in result.get("testActionOutput", []):
                uri = output.get("uri")
                if not uri:
                    continue
                parsed = urlsplit(uri)
                if parsed.scheme != "file" or parsed.netloc not in ("", "localhost"):
                    raise ValueError("BEP outputs require local file URIs")
                path = Path(unquote(parsed.path))
                # Resolve parent aliases without following the declared output itself.
                relative = (path.parent.resolve(strict=True) / path.name).relative_to(
                    source
                )
                files[relative] = path
                has_xml |= path.name == "test.xml"
            if result.get("status") == "PASSED" and not has_xml:
                raise ValueError("A passing test has no declared JUnit result")
    if exit_code == 0 and (not finished or not results):
        raise ValueError("Successful Bazel invocation has no complete test results")
    return results, files


def publish(workspace, events, exit_code, reports_root=None):
    workspace = Path(workspace).resolve(strict=True)
    link = workspace / "bazel-testlogs"
    if not link.is_symlink():
        raise ValueError("Bazel testlogs must be a symlink, not a directory")
    original = os.readlink(link)
    results, files = read_results(events, link.resolve(strict=True), exit_code)
    if reports_root is not None:
        reports_root = Path(reports_root).resolve()
        reports_root.mkdir(parents=True, exist_ok=True)
    destination = Path(
        tempfile.mkdtemp(
            prefix="bazel-invocation-testlogs.", dir=reports_root or workspace.parent
        )
    )
    temporary_link = workspace / (".bazel-testlogs-" + uuid.uuid4().hex)
    try:
        for relative, path in files.items():
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if path.is_dir():
                # Nested XML is not a JUnit result unless the BEP declares it.
                shutil.copytree(
                    path,
                    target,
                    dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("test.xml"),
                )
            else:
                shutil.copy2(path, target)
        shutil.copy2(events, destination / "build-events.json")
        (destination / "invocation-results.json").write_text(
            json.dumps({"bazel_exit_code": exit_code, "tests": results}, indent=2)
            + "\n"
        )
        if not link.is_symlink() or os.readlink(link) != original:
            raise RuntimeError("Bazel testlogs changed while collecting results")
        temporary_link.symlink_to(reports_root or destination, target_is_directory=True)
        os.replace(temporary_link, link)
    except BaseException:
        temporary_link.unlink(missing_ok=True)
        shutil.rmtree(destination)
        raise
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path)
    parser.add_argument("events", type=Path)
    parser.add_argument("exit_code", type=int)
    parser.add_argument("--reports-root", type=Path)
    args = parser.parse_args()
    print(publish(args.workspace, args.events, args.exit_code, args.reports_root))
