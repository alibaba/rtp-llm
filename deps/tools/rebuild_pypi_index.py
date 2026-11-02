#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rebuild the PEP 503 simple-* wheel indexes of oss://rtp-opensource.

Every generated index entry links directly to the canonical rtp_llm/<arch>/ wheel
objects (single source of truth); no simple/ duplicate object tree is produced.

Usage:
    python3 rebuild_pypi_index.py --dry-run
    python3 rebuild_pypi_index.py --prefix rtp_llm/simple-rebuild-test/   # scratch
    python3 rebuild_pypi_index.py            # default prefix rtp_llm/ (live indexes)

Requires ossutil (v2) on PATH with credentials already configured by ossutil itself
(e.g. ~/.ossutilconfig or OSS_* env vars). This script never handles credentials.
Idempotent: re-running regenerates identical pages; sha256 values are cached locally
(cache --cache path) so repeat runs do not re-download wheels.
"""

import argparse
import concurrent.futures
import datetime
import hashlib
import html
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import urllib.parse

BUCKET = "rtp-opensource"
REGION = "cn-hangzhou"
PUBLIC_BASE = "https://%s.oss-cn-hangzhou.aliyuncs.com" % BUCKET
SRC_ROOT = "rtp_llm"                 # wheels live at rtp_llm/<arch>/<file>.whl
DEFAULT_PREFIX = "rtp_llm/"          # index pages go to <prefix>simple-<platform>/...
CACHE_VERSION = 1

# arch dir under rtp_llm/ -> wheel platform-tag class -> simple-<platform> index trees.
#
# Mapping derived from the existing index set (simple, simple-arm, simple-cpu,
# simple-cuda12, simple-cuda12_9, simple-cuda12_arm, simple-cuda13, simple-cuda13_arm,
# simple-rocm) and the per-lane requirements files in deps/:
#   cu126 -> cuda12 lane, cu129 -> cuda12.9 lane, cu130 -> cuda13 lane,
#   cu130_arm -> cuda13 arm lane, rocm -> rocm lane, cpu -> cpu lane.
# cu129 aarch64 wheels go to simple-cuda12_arm: that is the only cuda12-family arm
# index and requirements_cuda12_arm.txt consumes cu126+cu129 aarch64 wheels.
# "any"-tagged (pure python) wheels are listed in both the base and the _arm tree of
# their arch family, matching the arm lanes which pin those same any-tagged wheels.
# Any other directory under rtp_llm/ (test/, utils/, eic/, ...) is skipped.
ARCH_PLATFORMS = {
    "cpu":       {"x86": ["simple-cpu"],        "arm": ["simple-cpu"],
                  "any": ["simple-cpu"]},
    "cu126":     {"x86": ["simple-cuda12"],     "arm": ["simple-cuda12_arm"],
                  "any": ["simple-cuda12", "simple-cuda12_arm"]},
    "cu129":     {"x86": ["simple-cuda12_9"],   "arm": ["simple-cuda12_arm"],
                  "any": ["simple-cuda12_9", "simple-cuda12_arm"]},
    "cu130":     {"x86": ["simple-cuda13"],     "arm": ["simple-cuda13_arm"],
                  "any": ["simple-cuda13", "simple-cuda13_arm"]},
    "cu130_arm": {"x86": [],                    "arm": ["simple-cuda13_arm"],
                  "any": ["simple-cuda13_arm"]},
    "rocm":      {"x86": ["simple-rocm"],       "arm": ["simple-rocm"],
                  "any": ["simple-rocm"]},
}

DEFAULT_CACHE = os.path.join("~", ".cache", "rtp-opensource", "rebuild_pypi_index.sha256.json")

HEX64 = re.compile(r"^[0-9a-fA-F]{64}$")


def log(msg):
    print(msg)
    sys.stdout.flush()


def human_size(n):
    value = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024 or unit == "TiB":
            return ("%d B" % n) if unit == "B" else ("%.2f %s" % (value, unit))
        value /= 1024.0


def normalize_prefix(prefix):
    p = prefix.strip()
    if not p:
        raise SystemExit("error: --prefix must not be empty")
    if p.startswith("/"):
        raise SystemExit("error: --prefix must be relative to the bucket root (no leading '/')")
    return p if p.endswith("/") else p + "/"


# --------------------------------------------------------------------------- ossutil plumbing

def run_ossutil(ossutil, region, args):
    cmd = [ossutil] + list(args) + ["--region", region]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          universal_newlines=True)
    if proc.returncode != 0:
        raise RuntimeError("ossutil failed (%d): %s\n%s" % (proc.returncode, " ".join(cmd),
                                                            (proc.stderr or "").strip()))
    return proc.stdout


def parse_ls(text):
    """Parse `ossutil ls` rows into object dicts.

    Rows look like: <mtime tokens> <size> <storage-class> <etag> oss://bucket/key
    Parsing is anchored on the trailing columns so the locale/timezone prefix
    (e.g. '2026-10-28 01:23:14 +0800 CST') never matters."""
    objs = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5 or not parts[-1].startswith("oss://"):
            continue
        try:
            size = int(parts[-4])
        except ValueError:
            continue
        objs.append({
            "url": parts[-1],
            "etag": parts[-2].strip('"'),
            "size": size,
            "mtime_raw": " ".join(parts[:-4]),
        })
    return objs


def oss_key(url):
    return url[len("oss://%s/" % BUCKET):]


def list_objects(ossutil, region, prefix):
    """Recursive listing of oss://BUCKET/<prefix> (prefix may end with '/' or not)."""
    out = run_ossutil(ossutil, region, ["ls", "oss://%s/%s" % (BUCKET, prefix)])
    objs = []
    for o in parse_ls(out):
        o["key"] = oss_key(o["url"])
        objs.append(o)
    return objs


def list_children(ossutil, region, prefix):
    """First-level entries (dirs and loose objects) under oss://BUCKET/<prefix>.

    Note: `ossutil ls -d` needs the prefix to end with '/' to list the children;
    without it the command returns just the prefix itself."""
    out = run_ossutil(ossutil, region,
                      ["ls", "oss://%s/%s/" % (BUCKET, prefix.strip("/")), "-d"])
    base = "oss://%s/%s" % (BUCKET, prefix.strip("/"))
    children = set()
    for line in out.splitlines():
        parts = line.split()
        if not parts:
            continue
        u = parts[-1]
        if u == base or u.startswith(base + "/"):
            rest = u[len(base):].strip("/")
            if rest:
                children.add(rest.split("/")[0])
    return children


# --------------------------------------------------------------------------- wheel helpers

def normalize_pkg(name):
    """PEP 503 project-name normalization: lowercase, runs of [-_.] -> '-'."""
    return re.sub(r"[-_.]+", "-", name).lower()


def wheel_pkg(fname):
    stem = fname[:-4] if fname.endswith(".whl") else fname
    return normalize_pkg(stem.split("-", 1)[0])


def wheel_platform_tag(fname):
    stem = fname[:-4] if fname.endswith(".whl") else fname
    return stem.rsplit("-", 1)[-1]


def classify_tag(tag):
    t = tag.lower()
    if "aarch64" in t or "arm64" in t or t.endswith("_arm") or t.startswith("arm"):
        return "arm"
    if "x86_64" in t or "amd64" in t or "x86" in t or "i686" in t:
        return "x86"
    return "any"


def parse_mtime(raw):
    """'2026-10-28 01:23:14 +0800 CST' -> comparable datetime (zone dropped on
    purpose: all rows of one listing share the same zone)."""
    tokens = raw.split()
    if len(tokens) >= 2:
        try:
            return datetime.datetime.strptime(" ".join(tokens[:2]), "%Y-%m-%d %H:%M:%S")
        except ValueError:
            pass
    return datetime.datetime.min


def collect_wheels(ossutil, region, arch, recursive):
    """Scan one arch dir. Returns (kept, skipped_events, stats).

    kept: canonical wheel dicts; skipped_events: (reason, wheel) tuples, e.g. the
    %-encoded duplicate keys (rocm holds pairs like 'torch-2.9.1%2Bgit...whl' next to
    the identical-bytes 'torch-2.9.1+git...whl')."""
    objs = list_objects(ossutil, region, "%s/%s/" % (SRC_ROOT, arch))
    wheels = []
    non_whl = 0
    subdir_wheels = 0
    for o in objs:
        if o["key"].endswith("/"):        # simulated-directory marker objects
            continue
        rel = o["key"][len("%s/%s/" % (SRC_ROOT, arch)):]
        fname = rel.rsplit("/", 1)[-1]
        if not fname.endswith(".whl"):
            non_whl += 1
            continue
        if "/" in rel and not recursive:
            subdir_wheels += 1
            continue
        w = dict(o)
        w["arch"] = arch
        w["fname"] = fname                          # literal name inside the object key
        w["decoded"] = urllib.parse.unquote(fname)  # the real wheel file name
        w["rel"] = rel                              # path under the arch dir (subdirs with --recursive)
        w["mtime"] = parse_mtime(o["mtime_raw"])
        wheels.append(w)

    groups = {}
    for w in wheels:
        groups.setdefault(w["decoded"], []).append(w)
    kept, skipped = [], []
    for name, ws in sorted(groups.items()):
        if len(ws) == 1:
            kept.append(ws[0])
            continue
        if len(set(w["etag"] for w in ws)) == 1:
            # same bytes under differently-encoded keys: prefer the key whose literal
            # file name already is the decoded name (no %-escapes baked into the key)
            canonical = [w for w in ws if w["fname"] == name]
            pick = canonical[0] if canonical else sorted(ws, key=lambda w: w["key"])[0]
            for w in ws:
                if w is not pick:
                    skipped.append(("duplicate-encoded-key", w))
        else:
            # same decoded name, different bytes: keep the most recently uploaded
            pick = sorted(ws, key=lambda w: (w["mtime"], w["key"]))[-1]
            for w in ws:
                if w is not pick:
                    skipped.append(("conflicting-content-kept-newest", w))
        kept.append(pick)
    return kept, skipped, {"non_whl": non_whl, "subdir_wheels": subdir_wheels}


# --------------------------------------------------------------------------- hashing

class HashStore(object):
    """sha256 per wheel object.

    Priority: local cache -> ossutil stat (x-oss-meta-sha256 user meta) -> download
    with ossutil cp and hash the bytes. No credentials involved here."""

    def __init__(self, ossutil, region, cache_path, workdir):
        self.ossutil = ossutil
        self.region = region
        self.cache_path = os.path.expanduser(cache_path)
        self.workdir = workdir
        self.lock = threading.Lock()
        self.cache = {"version": CACHE_VERSION, "objects": {}}
        self.sources = {"cache": 0, "meta": 0, "download": 0}
        self.download_bytes = 0
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path) as fh:
                    data = json.load(fh)
                if data.get("version") == CACHE_VERSION:
                    self.cache = data
            except (ValueError, OSError):
                log("warning: cache file %s unreadable, starting fresh" % self.cache_path)

    def save(self):
        with self.lock:
            tmp = self.cache_path + ".tmp"
            d = os.path.dirname(self.cache_path)
            if d and not os.path.isdir(d):
                os.makedirs(d)
            with open(tmp, "w") as fh:
                json.dump(self.cache, fh, indent=1, sort_keys=True)
            os.replace(tmp, self.cache_path)

    def _meta_sha256(self, key):
        out = run_ossutil(self.ossutil, self.region,
                          ["stat", "oss://%s/%s" % (BUCKET, key)])
        for line in out.splitlines():
            if ":" not in line:
                continue
            k, v = line.split(":", 1)
            if re.search(r"x-oss-meta[-_]?sha256", k, re.I):
                v = v.strip()
                if HEX64.match(v):
                    return v.lower()
        return None

    def _download_sha256(self, w, idx):
        local = os.path.join(self.workdir, "hash-%06d.whl" % idx)
        try:
            run_ossutil(self.ossutil, self.region,
                        ["cp", "oss://%s/%s" % (BUCKET, w["key"]), local, "-f"])
            h = hashlib.sha256()
            with open(local, "rb") as fh:
                for chunk in iter(lambda: fh.read(4 * 1024 * 1024), b""):
                    h.update(chunk)
            return h.hexdigest()
        finally:
            if os.path.exists(local):
                os.unlink(local)

    def hash_wheel(self, w, idx):
        key = w["key"]
        with self.lock:
            cached = self.cache["objects"].get(key)
        if cached and cached.get("etag") == w["etag"] and cached.get("size") == w["size"] \
                and HEX64.match(cached.get("sha256", "")):
            with self.lock:
                self.sources["cache"] += 1
            return cached["sha256"]

        sha = self._meta_sha256(key)
        source = "meta"
        if sha is None:
            sha = self._download_sha256(w, idx)
            source = "download"
            with self.lock:
                self.download_bytes += w["size"]
        with self.lock:
            self.sources[source] += 1
            self.cache["objects"][key] = {
                "etag": w["etag"], "size": w["size"], "sha256": sha, "source": source,
            }
        return sha


# --------------------------------------------------------------------------- page rendering

PAGE_TMPL = u"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="pypi:repository-version" content="1.0">
  <title>{title}</title>
</head>
<body>
<!-- generated by deps/tools/rebuild_pypi_index.py; every link points at the canonical
     oss://{bucket}/{src}/<arch>/ objects (single source of truth, no duplicate tree) -->
{links}
</body>
</html>
"""


def render_pkg_page(pkg, entries):
    lines = [u'  <a href="%s">%s</a><br>' % (href, html.escape(fname))
             for fname, href in entries]
    return PAGE_TMPL.format(title=u"Links for %s" % pkg, links=u"\n".join(lines),
                            bucket=BUCKET, src=SRC_ROOT).encode("utf-8")


def render_root_page(pkgs):
    lines = [u'  <a href="%s/">%s</a><br>' % (p, p) for p in pkgs]
    return PAGE_TMPL.format(title=u"Simple index", links=u"\n".join(lines),
                            bucket=BUCKET, src=SRC_ROOT).encode("utf-8")


# --------------------------------------------------------------------------- main

def build_parser():
    p = argparse.ArgumentParser(
        description="Rebuild PEP 503 simple indexes for oss://%s/%s/<arch>/ wheels; "
                    "index pages link directly to the arch objects." % (BUCKET, SRC_ROOT))
    p.add_argument("--dry-run", action="store_true",
                   help="print the plan only, no hashing and no upload")
    p.add_argument("--prefix", default=DEFAULT_PREFIX,
                   help="OSS prefix under which the simple-<platform>/ trees are written "
                        "(default: %(default)s = live indexes; use a scratch prefix such "
                        "as rtp_llm/simple-rebuild-test/ for validation)")
    p.add_argument("--recursive", action="store_true",
                   help="also index .whl files in subdirectories of an arch dir (dated "
                        "build snapshots); by default only top-level wheels are indexed, "
                        "matching what the requirements files consume")
    p.add_argument("--jobs", type=int, default=4,
                   help="parallel hash downloads (default: %(default)s)")
    p.add_argument("--cache", default=DEFAULT_CACHE,
                   help="local sha256 cache file (default: %(default)s)")
    p.add_argument("--workdir", default=None,
                   help="scratch dir for downloads and generated pages (default: a fresh "
                        "temp dir, removed afterwards)")
    p.add_argument("--ossutil", default=None,
                   help="path to ossutil (default: PATH lookup)")
    p.add_argument("--region", default=REGION, help="OSS region (default: %(default)s)")
    p.add_argument("--prune", action="store_true",
                   help="also delete <prefix>/<platform>/<pkg>/ trees of packages that "
                        "are not part of the rebuilt plan (default: off, stale pages are "
                        "left in place)")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    ossutil = args.ossutil or shutil.which("ossutil")
    if not ossutil:
        raise SystemExit("error: ossutil not found on PATH (pass --ossutil); credentials "
                         "must be configured via ossutil itself (e.g. ~/.ossutilconfig)")
    prefix = normalize_prefix(args.prefix)

    log("bucket: oss://%s   region: %s   ossutil: %s" % (BUCKET, args.region, ossutil))
    log("wheels scanned at: oss://%s/<arch>/   (single source of truth)" % SRC_ROOT)
    log("index destination: oss://%s/%ssimple-<platform>/" % (BUCKET, prefix))
    if prefix == DEFAULT_PREFIX and not args.dry_run:
        log("NOTE: default prefix writes over the LIVE rtp_llm/simple-* indexes")
    log("")

    # ---- discover the first level of rtp_llm/
    children = list_children(ossutil, args.region, SRC_ROOT)
    mapped = sorted(c for c in children if c in ARCH_PLATFORMS)
    simple_trees = sorted(c for c in children if c.startswith("simple"))
    loose = sorted(c for c in children
                   if c not in ARCH_PLATFORMS and not c.startswith("simple") and "." in c)
    unknown = sorted(c for c in children
                     if c not in ARCH_PLATFORMS and not c.startswith("simple") and "." not in c)

    plan = {}           # platform -> {pkg: [(fname, href, key), ...]}
    hash_targets = {}   # key -> wheel (needs exactly one sha256)
    total_bytes = 0
    skipped_events = []

    log("== source scan: oss://%s/ ==" % SRC_ROOT)
    for arch in mapped:
        kept, skipped, stats = collect_wheels(ossutil, args.region, arch, args.recursive)
        skipped_events.extend(skipped)
        subdir_note = ""
        if stats["subdir_wheels"]:
            subdir_note = (", %d in subdirs (not indexed; use --recursive to include)"
                           % stats["subdir_wheels"])
        log("  arch %-9s: %d top-level wheel(s)%s, %d non-whl object(s) skipped"
            % (arch, len(kept), subdir_note, stats["non_whl"]))
        for w in kept:
            tag = wheel_platform_tag(w["decoded"])
            tag_class = classify_tag(tag)
            if tag_class == "any" and tag != "any":
                log("    warning: unrecognized platform tag %r on %s, treated as "
                    "arch-independent" % (tag, w["decoded"]))
            platforms = ARCH_PLATFORMS[arch].get(tag_class)
            if not platforms:
                log("    warning: %s wheel %s has no target platform in arch %s, skipped"
                    % (tag_class, w["decoded"], arch))
                continue
            href = "%s/%s/%s/%s" % (PUBLIC_BASE, SRC_ROOT, arch,
                                    urllib.parse.quote(w["rel"]))
            entry_added = False
            for plat in platforms:
                pkg = wheel_pkg(w["decoded"])
                entries = plan.setdefault(plat, {}).setdefault(pkg, [])
                dup = [e for e in entries if e[0] == w["decoded"]]
                if dup:
                    if hash_targets.get(dup[0][2], {}).get("etag") == w["etag"]:
                        # identical bytes already listed from another arch dir (e.g.
                        # fast_hadamard_transform exists in cu126 and cu129); a single
                        # entry per file name keeps the page unambiguous for pip
                        log("    note: %s has identical copies in two arch dirs; one "
                            "entry kept in %s/%s" % (w["decoded"], plat, pkg))
                        continue
                    log("    warning: %s listed twice in %s/%s with different bytes "
                        "(same file name from different arch dirs)"
                        % (w["decoded"], plat, pkg))
                entries.append((w["decoded"], href, w["key"]))
                entry_added = True
            if entry_added and w["key"] not in hash_targets:
                hash_targets[w["key"]] = w
                total_bytes += w["size"]

    for d in unknown:
        objs = list_objects(ossutil, args.region, "%s/%s/" % (SRC_ROOT, d))
        whls = [o for o in objs if o["key"].endswith(".whl")]
        log("  dir  %-9s: not mapped, skipped (%d object(s), %d whl)"
            % (d, len(objs), len(whls)))
    if loose:
        log("  loose objects directly under %s/: %d (not indexed): %s"
            % (SRC_ROOT, len(loose),
               ", ".join(loose[:6]) + (" ..." if len(loose) > 6 else "")))
    if simple_trees:
        log("  existing simple* index trees left untouched: %s" % ", ".join(simple_trees))
    for reason, w in skipped_events:
        log("  skipped (%s): %s" % (reason, w["key"]))
    log("")

    log("== platform plan ==")
    for plat in sorted(plan):
        pkgs = plan[plat]
        links = sum(len(v) for v in pkgs.values())
        log("  %-18s: %d package(s), %d file link(s)" % (plat, len(pkgs), links))
    if not plan:
        raise SystemExit("error: nothing to index (no wheels found under mapped arch dirs)")
    log("")
    log("== totals ==")
    log("  %d canonical wheel(s), %s; sha256 via ossutil stat meta, else download "
        "(local cache: %s)" % (len(hash_targets), human_size(total_bytes), args.cache))
    log("")

    if args.dry_run:
        log("dry-run: no hashing, no upload. Pages would be written to:")
        for plat in sorted(plan):
            log("  %s/%s%s/" % (PUBLIC_BASE, prefix, plat))
        return 0

    # ---- hashing
    own_tmp = args.workdir is None
    workdir = args.workdir or tempfile.mkdtemp(prefix="rebuild_pypi_index_")
    if not os.path.isdir(workdir):
        os.makedirs(workdir)
    store = HashStore(ossutil, args.region, args.cache, workdir)
    sha_by_key = {}
    log("hashing %d wheel(s) ..." % len(hash_targets))
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
            futures = {pool.submit(store.hash_wheel, w, i): w
                       for i, w in enumerate(hash_targets.values())}
            done = 0
            for fut in concurrent.futures.as_completed(futures):
                w = futures[fut]
                sha_by_key[w["key"]] = fut.result()
                done += 1
                log("  [%3d/%3d] %s  %s" % (done, len(hash_targets),
                                            sha_by_key[w["key"]][:16] + "...", w["key"]))
    finally:
        try:
            store.save()
        except Exception as exc:  # keep the original error visible if saving fails
            log("warning: could not save hash cache: %s" % exc)
    log("hash sources: %s (re-downloaded %s)"
        % (store.sources, human_size(store.download_bytes)))
    log("")

    # ---- render pages
    outdir = os.path.join(workdir, "index_out")
    if os.path.isdir(outdir):
        shutil.rmtree(outdir)
    for plat, pkgs in sorted(plan.items()):
        plat_dir = os.path.join(outdir, plat)
        for pkg, entries in sorted(pkgs.items()):
            pkg_dir = os.path.join(plat_dir, pkg)
            os.makedirs(pkg_dir)
            filled = [(fname, href + "#sha256=" + sha_by_key[key])
                      for fname, href, key in entries]
            with open(os.path.join(pkg_dir, "index.html"), "wb") as fh:
                fh.write(render_pkg_page(pkg, sorted(filled)))
        os.makedirs(plat_dir, exist_ok=True)
        with open(os.path.join(plat_dir, "index.html"), "wb") as fh:
            fh.write(render_root_page(sorted(pkgs)))
    log("rendered index pages under %s" % outdir)
    log("")

    # ---- upload
    log("uploading ...")
    for plat in sorted(plan):
        src_dir = os.path.join(outdir, plat) + "/"
        dest = "oss://%s/%s%s/" % (BUCKET, prefix, plat)
        run_ossutil(ossutil, args.region,
                    ["cp", "-r", src_dir, dest, "-f",
                     "--content-type", "text/html; charset=utf-8"])
        log("  uploaded %s%s/" % (prefix, plat))

    if args.prune:
        log("")
        log("pruning stale package trees ...")
        for plat in sorted(plan):
            existing = list_children(ossutil, args.region, "%s%s" % (prefix, plat))
            stale = [c for c in existing if c not in plan[plat] and "." not in c]
            for pkg in sorted(stale):
                run_ossutil(ossutil, args.region,
                            ["rm", "-r", "-f", "oss://%s/%s%s/%s/" % (BUCKET, prefix, plat, pkg)])
                log("  removed %s%s/%s/" % (prefix, plat, pkg))
            if not stale:
                log("  %s: nothing stale" % plat)

    log("")
    log("done. index roots:")
    for plat in sorted(plan):
        log("  %s/%s%s/" % (PUBLIC_BASE, prefix, plat))
    if own_tmp:
        shutil.rmtree(workdir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
