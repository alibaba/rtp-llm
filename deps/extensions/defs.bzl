# The single extension for all non-BCR repos: archive repos (http_deps/git_deps), cc_view
# C++ archives derived from deps.json, cuda/rocm/python autoconfig, and the
# rtp_extension/arch_config injection points gated by RTP_INTERNAL_SOURCE (without that env
# an open-source clone falls back to the open-source stubs; mctx.getenv records the env in
# the invalidation key). Internal payloads (PPU pip hub + torch/rpm/git archives) are created
# in the same env branch, so siblings reference them by apparent name without root use_repo.
#
# Red line: URL/sha/pip-index credentials are read [only] via mctx.read from
# <RTP_INTERNAL_SOURCE>/deps/ppu.json. This file has zero intranet direct links and zero
# private names, mechanism only. An open-source clone creates none of these repos.
load("//deps/extensions:http_deps.bzl", "HTTP_DEPS_EXPORTS", "http_deps")
load("//deps/extensions:git_deps.bzl", "GIT_DEPS_EXPORTS", "git_deps")
load("@rtp_llm//3rdparty/cuda_config:cuda_configure.bzl", "cuda_configure")
load("@rtp_llm//3rdparty/gpus:rocm_configure.bzl", "rocm_configure")
load("@rtp_llm//3rdparty/py:python_configure.bzl", "python_configure")
load("@rtp_llm//:release_version.bzl", "read_release_version")
load("@bazel_tools//tools/build_defs/repo:local.bzl", "local_repository", "new_local_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive", "http_file")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")
load("@rules_python//python/private/pypi:whl_library.bzl", "whl_library")

# Path segments, relative to the internal_source root, of the payload data. The lock path is
# profile-specific and is read from the same private manifest below.
_DATA_SEGMENTS = ["deps", "ppu.json"]

def _internal_path_segments(path):
    if not path.startswith("internal_source/"):
        fail("private profile lock path must be repo-root relative under internal_source/: " + path)
    return path[len("internal_source/"):].split("/")

def _cc_view_repos(mctx):
    # The same cc_view record feeds both the wheel-consuming side and the C++ archive repo
    # created here — a single coordinate.
    # cc_view entries are all wheels, so type=zip is hardcoded; a record with the same name is created only once.
    manifest = json.decode(mctx.read(Label("@rtp_llm//deps:deps.json")))
    seen = {}
    for pkg in manifest["packages"]:
        for _profile, info in pkg.get("per_profile", {}).items():
            cv = info.get("cc_view")
            if not cv:
                continue
            if cv["name"] in seen:
                continue
            seen[cv["name"]] = True
            http_archive(
                name = cv["name"],
                sha256 = cv["sha256"],
                urls = cv["urls"],
                type = "zip",
                build_file = cv["build_file"],
            )
    return sorted(seen.keys())

# ---- Real PPU pip hub (mirrors the internals of the rules_python 0.33.2 pip extension) -----
def _normalize(name):
    # rules_python normalize_name (lowercase + [-.]→_ + collapse separators), plus
    # stripping any [extras] suffix first: hub package keys are plain distribution names.
    name = name.split("[", 1)[0]
    name = name.replace("-", "_").replace(".", "_").lower()
    if "__" not in name:
        return name
    return "_".join([p for p in name.split("_") if p])

def _flush_pending(pending):
    """Return a lock requirement only when its artifact identity is complete."""
    norm, body, hashes = pending
    if not hashes:
        fail("private lock entry is missing --hash: " + body)
    return (norm, body + " " + " ".join(hashes))

def _parse_ppu_lock(content):
    """Extract the pin lines of the lock; returns [(normalized, requirement), ...].

    Both forms must be recognized: ``name==version`` and PEP 508 direct URLs ``name @ url`` —
    self-built artifacts are pinned by direct URL (the same version on the public net is a
    different set of bytes). ``--hash`` continuation lines must be collected and appended back
    onto the requirement: pip only enforces verification when it sees --hash in the
    requirement; dropping it means an unverified download (especially bad for direct URLs).

    Skips blank lines, comments, and pip option lines while preserving environment markers.
    """
    pkgs = []
    pending = None  # (normalized, requirement body, [hashes]) of the current pin
    for raw in content.replace("\r", "").split("\n"):
        if not raw:
            continue
        stripped = raw.strip()
        # Indented lines: --hash continuations belong to the current pin; the rest
        # (# via continuation comments) are ignored.
        if raw[0] == " " or raw[0] == "\t":
            if pending != None and stripped.startswith("--hash="):
                pending[2].append(stripped.split(" ")[0].rstrip("\\").strip())
            continue
        if raw[0] == "#":
            continue
        if raw[0] == "-":
            # A flush-left --hash cannot occur (pip-compile always indents); only option
            # lines like --index-url appear here.
            continue
        line = raw.strip()
        if line.endswith("\\"):
            line = line[:-1].strip()
        requirement, marker_separator, marker = line.partition(";")
        line = requirement.strip()
        marker = marker.strip() if marker_separator else ""
        body = None
        norm = None
        head, sep, tail = line.partition("==")
        if sep:
            name = head.strip()
            version = tail.strip().split(" ")[0].strip()
            if name and version:
                norm = _normalize(name)
                body = "{}=={}".format(name, version)
        else:
            head, sep, tail = line.partition("@")
            if sep:
                name = head.strip()
                url = tail.strip().split(" ")[0].strip()
                if name and url:
                    norm = _normalize(name)
                    body = "{} @ {}".format(name, url)
        if body != None and marker:
            body += " ; " + marker
        if body == None:
            continue
        if pending != None:
            pkgs.append(_flush_pending(pending))
        pending = (norm, body, [])
    if pending != None:
        pkgs.append(_flush_pending(pending))
    return pkgs

def _ppu_hub_impl(rctx):
    names = rctx.attr.packages
    prefix = rctx.attr.repo_prefix

    rctx.file(
        "BUILD.bazel",
        "package(default_visibility = [\"//visibility:public\"])\n" +
        "exports_files([\"requirements.bzl\"])\n",
    )

    # requirements.bzl is the only surface consumed by arch_select.bzl.
    # Uses canonical direct addressing @@<canonical>//pkg:pkg: under bzlmod, rctx.attr.name is
    # already the canonical name; spelling it as "@<name>" would look it up as an apparent name
    # in the [consumer's] repo mapping and be necessarily invisible.
    # @@ bypasses the mapping — the hub is an internal implementation detail of the extension
    # and should not require root use_repo.
    reqs_bzl = """\
\"\"\"Private pip hub folded into the rtp_non_module_deps extension.\"\"\"

_HUB = "{hub}"

def _clean(name):
    name = name.replace("-", "_").replace(".", "_").lower()
    if "__" not in name:
        return name
    return "_".join([p for p in name.split("_") if p])

def requirement(name):
    return "@@" + _HUB + "//" + _clean(name) + ":pkg"
""".format(hub = rctx.attr.name)
    rctx.file("requirements.bzl", reqs_bzl)

    # One alias directory per package: pkg → @pip_ppu_torch_<norm>//:pkg.
    # (spokes within the same extension are visible by apparent name, so the alias actual resolves).
    for n in names:
        spoke = "@{}{}".format(prefix, n)
        rctx.file(
            n + "/BUILD.bazel",
            "package(default_visibility = [\"//visibility:public\"])\n" +
            "alias(name = \"pkg\", actual = \"{spoke}//:pkg\")\n".format(spoke = spoke),
        )

_ppu_hub_repository = repository_rule(
    implementation = _ppu_hub_impl,
    attrs = {
        "packages": attr.string_list(mandatory = True),
        "repo_prefix": attr.string(mandatory = True),
    },
)

# ---- Internal view: real payloads (credentials read only via mctx.read from internal_source) ----
def _join(base, segments):
    p = base
    for seg in segments:
        p = p.get_child(seg)
    return p

def _ppu_pip_args(index):
    # whl_library uses pip, which already aggregates candidates across index-url + all
    # extra-index-urls and picks the best version, so do not pass --index-strategy (that is a
    # uv-only flag; pip would fail with "no such option").
    args = ["--index-url=" + index["index_url"]]
    for extra in index.get("extra_index_urls", []):
        args.append("--extra-index-url=" + extra)
    return args

def _build_ppu_payload(mctx, internal_path):
    # internal_path is the mctx.path of the internal_source root; credentials are read only here via mctx.read.
    data = json.decode(mctx.read(_join(internal_path, _DATA_SEGMENTS)))
    artifacts = data["artifacts"]

    # torch C++ archives: name/URL/sha/build_file all come from ppu.json (the open-source side
    # contains no private names or direct links).
    for t in artifacts["torch"]:
        http_archive(
            name = t["name"],
            sha256 = t["sha256"],
            urls = t["urls"],
            type = t["type"],
            build_file = t["build_file"],
        )

    # Intranet rpms: consumption surface @<name>//file:file (see internal_source/deps/3rdparty/*/BUILD).
    for r in artifacts["rpm"]:
        http_file(
            name = r["name"],
            urls = r["urls"],
            sha256 = r["sha256"],
        )

    # Internal local source trees: paths relative to the internal_source root. Must be
    # standalone repos — they carry their own "//:" root-relative labels, and the open-source
    # .bazelignore ignores internal_source/rdma.
    for loc in artifacts.get("local", []):
        new_local_repository(
            name = loc["name"],
            path = str(internal_path.get_child(loc["path"])),
            build_file = loc["build_file"],
        )

    # PPU C++ git repos (PPU branches of flashinfer/flashmla): the intranet gitlab has no
    # tarball channel, so they stay in git form. remote/commit/build_file/patches all come from ppu.json.
    for g in artifacts.get("git", []):
        new_git_repository(
            name = g["name"],
            remote = g["remote"],
            commit = g["commit"],
            build_file = g["build_file"],
            patches = g.get("patches", []),
        )

    # Every private profile has an independent lock and pip hub. PPU remains one profile, while
    # CUDA13/ARM consume the same private supply mechanism without routing through another arch's
    # wheels. The lock itself is the package set, so no second package list is maintained here.
    for profile_name in sorted(data["profiles"].keys()):
        profile = data["profiles"][profile_name]
        hub = profile["hub"]
        spoke_prefix = hub + "_"
        index = profile["index"]
        pip_args = _ppu_pip_args(index)
        lock_content = mctx.read(_join(internal_path, _internal_path_segments(profile["lock"])))
        pkgs = _parse_ppu_lock(lock_content)
        for norm, req in pkgs:
            whl_library(
                name = spoke_prefix + norm,
                requirement = req,
                repo = hub,
                repo_prefix = spoke_prefix,
                python_interpreter = index["python_interpreter"],
                extra_pip_args = pip_args,
                timeout = 3600,
            )
        _ppu_hub_repository(
            name = hub,
            packages = [norm for norm, _ in pkgs],
            repo_prefix = spoke_prefix,
        )

def _rtp_non_module_deps_impl(mctx):
    http_deps()
    git_deps()
    cc_view_names = _cc_view_repos(mctx)
    cuda_configure(name = "local_config_cuda")
    rocm_configure(name = "local_config_rocm")
    python_configure(name = "local_config_python")
    read_release_version(name = "release_version")

    # The root module's direct-dependency list is declared by the extension itself;
    # `scripts/rtpcli bazel mod-tidy` uses it to maintain use_repo. Internal
    # payloads (PPU hub/spokes, torch/rpm archives) are [deliberately] not on the list: they
    # are referenced only by same-extension siblings via apparent name; root need not and
    # should not import them.
    direct_deps = HTTP_DEPS_EXPORTS + GIT_DEPS_EXPORTS + cc_view_names + [
        "local_config_cuda",
        "local_config_rocm",
        "local_config_python",
        "release_version",
        "rtp_extension",
        "arch_config",
    ]

    internal = mctx.getenv("RTP_INTERNAL_SOURCE", "")
    if internal:
        # The env is a [switch]. --repo_env does not expand %workspace%, so use the value
        # directly only when it is an absolute path; otherwise derive the internal_source
        # directory from the main repo root.
        if internal.startswith("/"):
            internal_path = mctx.path(internal)
        else:
            root = mctx.path(Label("@rtp_llm//deps:BUILD")).dirname.dirname
            internal_path = root.get_child("internal_source")
        local_repository(name = "rtp_extension", path = str(internal_path.get_child("deps")))
        local_repository(name = "arch_config", path = str(internal_path.get_child("bazel")))
        _build_ppu_payload(mctx, internal_path)
        # Sensitive-information red line: internal-view repo specs contain intranet index/bucket
        # URLs; reproducible=False would write them into MODULE.bazel.lock and leak them once
        # committed. Hence explicit reproducible=True ⇒ this extension as a whole stays out of
        # the lock (internal results are not reproducible across machines anyway).
        return mctx.extension_metadata(
            root_module_direct_deps = direct_deps,
            root_module_direct_dev_deps = [],
            reproducible = True,
        )
    else:
        # Open-source clone: only the open-source stubs land; no internal payloads are created.
        # This branch produces only public archives; reproducible=False ⇒ public repo specs
        # enter the lock normally (shareable/reproducible; the committed lock goes through this
        # branch, RTP_INTERNAL_SOURCE=null, zero intranet direct links).
        local_repository(name = "rtp_extension", path = "deps")
        local_repository(name = "arch_config", path = "arch_config")
        return mctx.extension_metadata(
            root_module_direct_deps = direct_deps,
            root_module_direct_dev_deps = [],
            reproducible = False,
        )

rtp_non_module_deps = module_extension(
    implementation = _rtp_non_module_deps_impl,
)
