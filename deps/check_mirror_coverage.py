#!/usr/bin/env python3
"""Mirror coverage: every declared object in our own bucket must actually exist.

This check is the **prerequisite** for "archives are allowed upstream same-bytes
fallback". With fallback candidates added, a missing bucket object no longer turns the
build red -- it silently goes upstream and the build stays green, so "forgot to publish
to the bucket" goes unnoticed. Move that worry from runtime to pre-commit: missing
objects hard-fail here.

Criterion: scan all URLs pointing at our own bucket in the declaration surface under
deps/extensions/, MODULE.bazel, and the JSON manifest consumed by the extension, then HEAD
each one.
  * 200            -> covered
  * 404            -> missing, FAIL (publish the object first, then commit the consumer)
  * other/timeout  -> WARN (a network fact, not a supply fact; do not block on incomplete evidence)

**Needs internet**, so it is not in the offline dependency gate; `scripts/rtpcli deps verify
--all --policy-only` runs it in CI, and it can also be executed standalone locally.

Usage: check_mirror_coverage.py <open-source repo root>
"""

import concurrent.futures as cf
import json
import os
import re
import sys
import urllib.error
import urllib.request

BUCKET = "https://rtp-opensource.oss-cn-hangzhou.aliyuncs.com"
BUCKET_RE = re.compile(
    r'"(https://rtp-opensource\.oss-[a-z0-9-]+\.aliyuncs\.com/[^"\s]+)"'
)
OWN_BUCKET_URL_RE = re.compile(
    r"^https://rtp-opensource\.oss-[a-z0-9-]+\.aliyuncs\.com/"
)
# Archives mostly no longer write literal URLs; they go through the two mirror.bzl
# helpers -- matching only literal strings would miss almost every archive, and "the scan
# surface silently missing real declarations" is worse than having no check. The two
# templates below must stay consistent with how mirror.bzl builds URLs.
MIRROR_CALL_RE = re.compile(r'rtp_mirror_urls\(\s*"([^"]+)"')
GITHUB_CALL_RE = re.compile(
    r'rtp_github_archive_urls\(\s*"([^"]+)"\s*,\s*"([^"]+)"\s*,\s*"([^"]+)"\s*\)'
)
SCAN_DIRS = ("deps/extensions",)
SCAN_FILES = ("MODULE.bazel", "deps/deps.json")
TIMEOUT = 30


def _urls_in(line):
    out = list(BUCKET_RE.findall(line))
    for path in MIRROR_CALL_RE.findall(line):
        out.append(BUCKET + "/" + path)
    for owner, repo, ref in GITHUB_CALL_RE.findall(line):
        out.append(
            "%s/archives/github.com/%s/%s/%s.tar.gz" % (BUCKET, owner, repo, ref)
        )
    return out


def _collect_json_urls(value, location, found):
    """Collect bucket URLs from a parsed manifest with stable structural locations."""
    if isinstance(value, dict):
        for key, child in value.items():
            _collect_json_urls(child, "%s.%s" % (location, key), found)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _collect_json_urls(child, "%s[%d]" % (location, index), found)
    elif isinstance(value, str) and OWN_BUCKET_URL_RE.match(value):
        found.setdefault(value, []).append(location)


def declared_urls(root):
    """Own-bucket URLs appearing in the declaration surface -> {url: [locations]}."""
    found = {}
    targets = [f for f in SCAN_FILES if os.path.isfile(os.path.join(root, f))]
    for rel in SCAN_DIRS:
        base = os.path.join(root, rel)
        for cur, _dirs, files in os.walk(base):
            for name in sorted(files):
                if name.endswith(".bzl"):
                    targets.append(os.path.relpath(os.path.join(cur, name), root))
    for rel in targets:
        path = os.path.join(root, rel)
        if rel == "deps/deps.json":
            with open(path, encoding="utf-8") as fh:
                _collect_json_urls(json.load(fh), rel, found)
            continue
        with open(path, encoding="utf-8", errors="replace") as fh:
            for lineno, line in enumerate(fh, 1):
                for url in _urls_in(line):
                    found.setdefault(url, []).append("%s:%d" % (rel, lineno))
    return found


def probe(url):
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            return resp.status
    except urllib.error.HTTPError as exc:
        return exc.code
    except (urllib.error.URLError, TimeoutError, OSError):
        # Only fold [network facts] into 0 (main() above records WARN without blocking).
        # All other exceptions propagate: this check is already a blocking CI step; if a
        # mistyped URL (ValueError) or a broken helper (AttributeError) were also folded
        # into 0, they would masquerade as "network unreachable" and pass -- exactly the
        # shape this check exists to prevent.
        return 0


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else ".."
    urls = declared_urls(root)
    if not urls:
        print(
            "OK: no own-bucket URLs in the declaration surface, coverage check unnecessary"
        )
        return 0
    ordered = sorted(urls)
    with cf.ThreadPoolExecutor(max_workers=12) as pool:
        status = dict(zip(ordered, pool.map(probe, ordered)))

    missing = [u for u in ordered if status[u] == 404]
    unknown = [u for u in ordered if status[u] not in (200, 404)]
    for url in unknown:
        print(
            "WARN unreachable (HTTP %s), cannot judge supply: %s  <- %s"
            % (status[url] or "timeout", url, ", ".join(urls[url]))
        )
    if missing:
        print(
            "FAIL: %d declared bucket objects do not exist (archives allow upstream fallback, so a missing"
            " object does not turn the build red, it only silently switches source -- must be caught here):"
            % len(missing)
        )
        for url in missing:
            print("  %s\n    declared at %s" % (url, ", ".join(urls[url])))
        print(
            "  Fix: publish the byte-identical object to the public OSS mirror with the configured "
            "OSS publisher, then commit the declaration consuming that URL"
        )
        return 1
    print(
        "OK: all %d declared bucket objects exist (%d declaration sites)"
        % (len(ordered), sum(len(v) for v in urls.values()))
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
