#!/usr/bin/env python3
"""Small public checkout entry point for Bazel and dependency maintenance.

This file intentionally has no dependency on the internal repository. The
public checkout owns its profile manifest and can therefore expose the same
command shape without importing private overlays or credentials.
"""

import argparse
import base64
import hashlib
import hmac
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import deque
from email.utils import formatdate
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
DEPS = ROOT / "deps"
BAZELISK = "bazelisk"
TAIL_LINES = 80


class UsageError(Exception):
    pass


class ExternalError(Exception):
    pass


OSS_BUCKET = "rtp-opensource"
OSS_ENDPOINT = "oss-cn-hangzhou.aliyuncs.com"
OSS_KEY_ID = "OSS_OPENSOURCE_KEY_ID"
OSS_KEY_SECRET = "OSS_OPENSOURCE_KEY_SECRET"


def _python():
    return sys.executable


def _profiles():
    manifest = json.loads((DEPS / "deps.json").read_text(encoding="utf-8"))
    return {item["name"]: item["bazel_config"] for item in manifest["profiles"]}


def _cache_root(profile):
    override = os.environ.get("BAZEL_CACHE_ROOT")
    base = (
        Path(override).expanduser()
        if override
        else Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "rtp-llm"
    )
    identity = hashlib.sha256(str(ROOT).encode("utf-8")).hexdigest()[:12]
    return base / (
        "bazel-%s-%s-%s" % (os.uname().machine, identity, profile or "default")
    )


def _precheck(profile=None):
    checks = []

    def add(name, ok, detail):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})

    bazelisk = shutil.which(BAZELISK)
    add("bazelisk_available", bool(bazelisk), bazelisk or "bazelisk is not on PATH")
    version_file = ROOT / ".bazelversion"
    rc_file = ROOT / ".bazeliskrc"
    expected = (
        version_file.read_text(encoding="utf-8").strip()
        if version_file.is_file()
        else ""
    )
    actual = ""
    if rc_file.is_file():
        for line in rc_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("USE_BAZEL_VERSION="):
                actual = line.split("=", 1)[1].strip()
                break
    add(
        "bazel_version_single_source",
        bool(expected and expected == actual),
        (
            "consistent (%s)" % expected
            if expected and expected == actual
            else ".bazelversion=%s .bazeliskrc=%s"
            % (expected or "missing", actual or "missing")
        ),
    )
    add(
        "module_bazel_present",
        (ROOT / "MODULE.bazel").is_file(),
        str(ROOT / "MODULE.bazel"),
    )
    cache = _cache_root(profile)
    if cache.exists():
        owner = cache.stat().st_uid
        current = os.geteuid()
        add(
            "cache_owner_matches",
            owner == current,
            (
                "uid=%d" % current
                if owner == current
                else "%s belongs to uid=%d, current uid=%d" % (cache, owner, current)
            ),
        )
    else:
        add("cache_owner_matches", True, "cache will be created by the current user")
    return {
        "ok": all(item["ok"] for item in checks),
        "checks": checks,
        "cache_root": str(cache),
    }


def _config_args(args):
    values = []
    for index, token in enumerate(args):
        if token == "--config" and index + 1 < len(args):
            values.append(args[index + 1])
        elif token.startswith("--config="):
            values.append(token.split("=", 1)[1])
    return values


def _profile_for(profile, args):
    profiles = _profiles()
    by_config = {config: name for name, config in profiles.items()}
    inferred = [by_config[value] for value in _config_args(args) if value in by_config]
    unique = list(dict.fromkeys(inferred))
    if profile and profile not in profiles:
        raise UsageError(
            "unknown profile %r (known: %s)" % (profile, ", ".join(profiles))
        )
    if profile and unique and unique != [profile]:
        raise UsageError(
            "--profile=%s conflicts with --config=%s" % (profile, unique[0])
        )
    return profile or (unique[0] if unique else None)


def _has_config(args, config):
    return config in _config_args(args)


def _split_options(args):
    profile = None
    stream = False
    batch = False
    log_file = None
    bazel_args = []
    index = 0
    while index < len(args):
        token = args[index]
        if token == "--":
            bazel_args.extend(args[index + 1 :])
            break
        if token == "--profile":
            if index + 1 >= len(args):
                raise UsageError("--profile requires a value")
            profile = args[index + 1]
            index += 2
            continue
        if token.startswith("--profile="):
            profile = token.split("=", 1)[1]
        elif token == "--stream":
            stream = True
        elif token == "--batch":
            batch = True
        elif token == "--log-file":
            if index + 1 >= len(args):
                raise UsageError("--log-file requires a path")
            log_file = args[index + 1]
            index += 1
        elif token.startswith("--log-file="):
            log_file = token.split("=", 1)[1]
        else:
            bazel_args.append(token)
        index += 1
    return profile, stream, batch, log_file, bazel_args


def _run_process(command, stream=False, log_file=None):
    tail = deque(maxlen=TAIL_LINES)
    log_handle = open(log_file, "w", encoding="utf-8") if log_file else None
    command_env = os.environ.copy()
    # The public entry point must not accidentally activate the internal Bzlmod
    # overlay when it is invoked from the monorepo checkout.
    command_env.pop("RTP_INTERNAL_SOURCE", None)

    def record(line):
        tail.append(line)
        if log_handle:
            log_handle.write(line)
            log_handle.flush()
        sys.stderr.write(line)
        sys.stderr.flush()

    try:
        if stream:
            process = subprocess.Popen(
                command,
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                env=command_env,
            )
            assert process.stdout is not None
            for line in process.stdout:
                record(line)
            rc = process.wait()
        else:
            with tempfile.TemporaryFile(mode="w+") as output:
                process = subprocess.Popen(
                    command,
                    cwd=str(ROOT),
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=command_env,
                )
                rc = process.wait()
                output.seek(0)
                for line in output:
                    record(line)
    finally:
        if log_handle:
            log_handle.close()
    return rc, "".join(tail)


def _emit(payload, emit_json):
    if emit_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        if payload.get("error"):
            print("ERROR: %s" % payload["error"], file=sys.stderr)
        elif payload.get("verb"):
            print("OK: %s" % payload["verb"])
        if payload.get("hints"):
            for hint in payload["hints"]:
                print("  %s" % hint)
    return 0 if payload.get("ok") else 1


def _oss_credentials():
    key_id = os.environ.get(OSS_KEY_ID, "").strip()
    key_secret = os.environ.get(OSS_KEY_SECRET, "").strip()
    if not key_id or not key_secret:
        raise UsageError(
            "writing to %s requires %s and %s"
            % (OSS_BUCKET, OSS_KEY_ID, OSS_KEY_SECRET)
        )
    return key_id, key_secret


def _oss_url(key):
    return "https://%s.%s/%s" % (OSS_BUCKET, OSS_ENDPOINT, quote(key, safe="/"))


def _oss_authorization(method, key, headers, key_id, key_secret):
    parts = [
        method,
        headers.get("Content-MD5", ""),
        headers.get("Content-Type", ""),
        headers["Date"],
    ]
    for name in sorted(name for name in headers if name.lower().startswith("x-oss-")):
        parts.append("%s:%s" % (name.lower(), headers[name]))
    parts.append("/%s/%s" % (OSS_BUCKET, key))
    digest = hmac.new(
        key_secret.encode("utf-8"), "\n".join(parts).encode("utf-8"), hashlib.sha1
    ).digest()
    return "OSS %s:%s" % (key_id, base64.b64encode(digest).decode("ascii"))


def _oss_request(method, key, key_id, key_secret, headers=None, data=None, timeout=30):
    sent = dict(headers or {})
    sent["Date"] = formatdate(usegmt=True)
    sent["Authorization"] = _oss_authorization(method, key, sent, key_id, key_secret)
    try:
        response = urlopen(
            Request(_oss_url(key), data=data, headers=sent, method=method),
            timeout=timeout,
        )
    except HTTPError as exc:
        body = exc.read(512)
        exc.close()
        if method == "HEAD" and exc.code == 404:
            return 404, {}, b""
        raise ExternalError(
            "OSS %s %s failed with HTTP %s: %s" % (method, key, exc.code, body[:200])
        )
    except URLError as exc:
        raise ExternalError("OSS %s %s is unreachable: %s" % (method, key, exc.reason))
    try:
        return response.status, dict(response.headers), response.read(512)
    finally:
        response.close()


def _oss_file_digests(path):
    sha256 = hashlib.sha256()
    md5 = hashlib.md5()
    size = 0
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1 << 20)
            if not chunk:
                break
            sha256.update(chunk)
            md5.update(chunk)
            size += len(chunk)
    return sha256.hexdigest(), md5.hexdigest(), size


def _oss_put(args, emit_json):
    parser = argparse.ArgumentParser(prog="rtpcli oss put")
    parser.add_argument("--repo", required=True, choices=["opensource"])
    parser.add_argument("--key", required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--yes", action="store_true")
    parser.add_argument("file")
    parsed = parser.parse_args(args)
    path = os.path.abspath(os.path.expanduser(parsed.file))
    if not os.path.isfile(path):
        raise UsageError("no such file: %s" % path)
    sha256, md5, size = _oss_file_digests(path)
    item = {
        "key": parsed.key,
        "path": path,
        "bytes": size,
        "sha256": sha256,
        "state": "planned",
        "url": _oss_url(parsed.key),
    }
    if parsed.dry_run:
        return _emit(
            {
                "ok": True,
                "verb": "oss put",
                "repo": parsed.repo,
                "applied": False,
                "objects": [item],
            },
            emit_json,
        )

    key_id, key_secret = _oss_credentials()
    status, remote_headers, _ = _oss_request(
        "HEAD", parsed.key, key_id, key_secret, timeout=30
    )
    if status == 200:
        remote_etag = remote_headers.get("ETag", "").strip('"')
        if remote_etag.lower() != md5.lower():
            raise ExternalError(
                "OSS object already exists with different content: %s" % parsed.key
            )
        return _emit(
            {
                "ok": True,
                "verb": "oss put",
                "repo": parsed.repo,
                "applied": False,
                "objects": [
                    {
                        "key": parsed.key,
                        "path": path,
                        "bytes": size,
                        "sha256": sha256,
                        "state": "already-uploaded",
                        "url": _oss_url(parsed.key),
                    }
                ],
            },
            emit_json,
        )
    item = {
        "key": parsed.key,
        "path": path,
        "bytes": size,
        "sha256": sha256,
        "state": "new",
        "url": _oss_url(parsed.key),
    }
    if parsed.yes:
        headers = {
            "Content-Type": "application/octet-stream",
            "Content-Length": str(size),
            "x-oss-forbid-overwrite": "true",
        }
        with open(path, "rb") as handle:
            _oss_request(
                "PUT",
                parsed.key,
                key_id,
                key_secret,
                headers=headers,
                data=handle,
                timeout=900,
            )
    return _emit(
        {
            "ok": True,
            "verb": "oss put",
            "repo": parsed.repo,
            "applied": bool(parsed.yes),
            "objects": [item],
        },
        emit_json,
    )


def _oss(args, emit_json):
    if not args or args[0] != "put":
        raise UsageError("usage: rtpcli oss put --repo opensource --key KEY --yes FILE")
    return _oss_put(args[1:], emit_json)


def _print_help():
    print("usage: scripts/rtpcli [--json] bazel|deps|oss ...")
    print("commands:")
    print(
        "  bazel  precheck, build, test, query, clean, mod-tidy, lock-check, lock-update"
    )
    print("  deps   profiles, sync, check, verify, gate")
    print("  oss    put --repo opensource --key KEY (--dry-run|--yes) FILE")


def _bazel_run(action, args, emit_json):
    profile, stream, batch, log_file, bazel_args = _split_options(args)
    if action in ("build", "test", "query", "clean"):
        profile = _profile_for(profile, bazel_args)
    if action in ("build", "test") and not profile:
        raise UsageError("bazel %s requires --profile or a known --config" % action)
    if profile:
        config = _profiles()[profile]
        if action in ("build", "test", "query") and not _has_config(bazel_args, config):
            bazel_args.insert(0, "--config=%s" % config)
    else:
        config = None

    precheck = _precheck(profile)
    if not precheck["ok"]:
        return _emit(
            {
                "ok": False,
                "verb": action,
                "profile": profile,
                "error": "Bazel precheck failed; Bazel was not started",
                "hints": [
                    item["detail"] for item in precheck["checks"] if not item["ok"]
                ],
                "precheck": precheck,
            },
            emit_json,
        )

    if action == "query":
        verb = "cquery" if profile else "query"
        action_tokens = [verb]
    elif action == "mod-tidy":
        verb = "mod-tidy"
        action_tokens = ["mod", "tidy"]
    elif action == "lock-check":
        verb = "lock-check"
        action_tokens = ["build"]
    elif action == "lock-update":
        verb = "lock-update"
        action_tokens = ["mod", "deps"]
    else:
        verb = action
        action_tokens = [action]
    command = [BAZELISK]
    if batch:
        command.append("--batch")
    command.extend(action_tokens)
    cache_root = precheck["cache_root"]
    if not any(
        token == "--disk_cache" or token.startswith("--disk_cache=")
        for token in bazel_args
    ):
        bazel_args.insert(0, "--disk_cache=%s" % cache_root)
    command.extend(bazel_args)
    rc, output_tail = _run_process(command, stream=stream, log_file=log_file)
    ok = rc in (0, 4)
    payload = {
        "ok": ok,
        "verb": verb,
        "profile": profile,
        "config": config,
        "rc": rc,
        "cmd": command,
        "cache_root": cache_root,
        "precheck": precheck,
        "output_tail": output_tail,
        "error": None if ok else "bazel %s failed (rc=%d)" % (verb, rc),
        "hints": [],
    }
    return _emit(payload, emit_json)


def _bazel_precheck(args, emit_json):
    profile, _, _, _, _ = _split_options(args)
    if profile:
        profile = _profile_for(profile, [])
    result = _precheck(profile)
    return _emit(
        {
            "ok": result["ok"],
            "verb": "precheck",
            "profile": profile,
            "precheck": result,
            "error": None if result["ok"] else "Bazel precheck failed",
            "hints": [item["detail"] for item in result["checks"] if not item["ok"]],
        },
        emit_json,
    )


def _lock_command(action, args, emit_json):
    if action == "lock-check":
        target = args[0] if args else "//deps:gen_wheel_requires"
        if len(args) > 1:
            raise UsageError("bazel lock-check accepts at most one target")
        return _bazel_run(
            "lock-check",
            ["--batch", "--stream", "--nobuild", "--lockfile_mode=error", target],
            emit_json,
        )
    if args:
        raise UsageError("bazel lock-update does not accept targets")
    return _bazel_run(
        "lock-update", ["--batch", "--stream", "--lockfile_mode=update"], emit_json
    )


def _run_external(command, emit_json):
    if not emit_json:
        return subprocess.call(command, cwd=str(ROOT))
    completed = subprocess.run(
        command, cwd=str(ROOT), text=True, capture_output=True, check=False
    )
    if completed.stdout:
        sys.stderr.write(completed.stdout)
    if completed.stderr:
        sys.stderr.write(completed.stderr)
    return completed.returncode


def _deps(args, emit_json):
    if not args:
        raise UsageError("usage: rtpcli deps profiles|sync|check|verify|gate")
    action, rest = args[0], list(args[1:])
    if action == "profiles":
        profiles = _profiles()
        payload = {
            "ok": True,
            "source": "deps/deps.json",
            "profiles": [
                {"profile": name, "bazel_config": config}
                for name, config in profiles.items()
            ],
            "error": None,
            "hints": [],
        }
        if emit_json:
            return _emit(payload, True)
        for item in payload["profiles"]:
            print("%-16s --config=%s" % (item["profile"], item["bazel_config"]))
        return 0

    parser = argparse.ArgumentParser(prog="rtpcli deps %s" % action)
    group = parser.add_mutually_exclusive_group(
        required=action in ("sync", "check", "verify")
    )
    group.add_argument("--profile", action="append")
    group.add_argument("--all", action="store_true")
    if action == "verify":
        parser.add_argument(
            "--policy-only",
            action="store_true",
            help="check declared public mirror objects without resolving private indexes",
        )
    parsed = parser.parse_args(rest)
    profiles = _profiles()
    if action == "sync":
        names = list(profiles) if parsed.all else parsed.profile
        if not names:
            raise UsageError("deps sync requires --profile NAME or --all")
        for name in names:
            if name not in profiles:
                raise UsageError("unknown profile %r" % name)
        command = [_python(), str(DEPS / "relock.py")]
        command += (
            ["--all"]
            if parsed.all
            else sum((["--profile", name] for name in names), [])
        )
        rc = _run_external(command, emit_json)
        return _emit(
            {
                "ok": rc == 0,
                "verb": "deps sync",
                "rc": rc,
                "error": None if rc == 0 else "dependency relock failed",
                "hints": [],
            },
            emit_json,
        )
    if action == "check":
        if parsed.profile:
            raise UsageError("deps check validates all public profiles; use --all")
        relock = _run_external(
            [_python(), str(DEPS / "relock.py"), "--all", "--check"], emit_json
        )
        if relock != 0:
            return _emit(
                {
                    "ok": False,
                    "verb": "deps check",
                    "rc": relock,
                    "error": "dependency relock check failed",
                    "hints": [],
                },
                emit_json,
            )
        rc = _run_external(["bash", str(DEPS / "gate.sh")], emit_json)
        return _emit(
            {
                "ok": rc == 0,
                "verb": "deps check",
                "rc": rc,
                "error": None if rc == 0 else "dependency gate failed",
                "hints": [],
            },
            emit_json,
        )
    if action == "verify":
        if not parsed.policy_only:
            raise UsageError(
                "public deps verify currently requires --policy-only; "
                "full index resolution is owned by the internal rtpcli"
            )
        if parsed.profile:
            raise UsageError("public deps verify supports --all only")
        rc = _run_external(
            [_python(), str(DEPS / "check_mirror_coverage.py"), str(ROOT)], emit_json
        )
        return _emit(
            {
                "ok": rc == 0,
                "verb": "deps verify",
                "scope": "all",
                "policy_only": True,
                "rc": rc,
                "error": None if rc == 0 else "public mirror coverage check failed",
                "hints": [],
            },
            emit_json,
        )
    if action == "gate":
        if rest:
            raise UsageError("deps gate takes no arguments")
        rc = _run_external(["bash", str(DEPS / "gate.sh")], emit_json)
        return _emit(
            {
                "ok": rc == 0,
                "verb": "deps gate",
                "rc": rc,
                "error": None if rc == 0 else "dependency gate failed",
                "hints": [],
            },
            emit_json,
        )
    raise UsageError("unknown deps command %r" % action)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    emit_json = False
    if argv and argv[0] == "--json":
        emit_json = True
        argv.pop(0)
    if not argv or argv in (["-h"], ["--help"]):
        _print_help()
        return 0
    if len(argv) < 2:
        raise UsageError("usage: scripts/rtpcli [--json] bazel|deps|oss ...")
    command, rest = argv[0], argv[1:]
    if command == "deps":
        return _deps(rest, emit_json)
    if command == "oss":
        return _oss(rest, emit_json)
    if command != "bazel":
        raise UsageError("unknown command %r" % command)
    action = rest[0]
    args = rest[1:]
    if action == "precheck":
        return _bazel_precheck(args, emit_json)
    if action in ("build", "test", "query", "clean", "mod-tidy"):
        return _bazel_run(action, args, emit_json)
    if action in ("lock-check", "lock-update"):
        return _lock_command(action, args, emit_json)
    raise UsageError("unknown bazel command %r" % action)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ExternalError as exc:
        print("[rtpcli] ERROR: %s" % exc, file=sys.stderr)
        raise SystemExit(1)
    except (OSError, ValueError, json.JSONDecodeError, UsageError) as exc:
        print("[rtpcli] ERROR: %s" % exc, file=sys.stderr)
        raise SystemExit(2)
