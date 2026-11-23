"""Discover BTC Bazel rule declarations in a checked-out revision and build a selected backend."""
import argparse
import json
import re
import subprocess
from pathlib import Path


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True)


def discover(repo):
    paths = git(repo, "ls-files", "-z").split("\0")
    for name in paths:
        p = Path(name)
        if p.name not in ("BUILD", "BUILD.bazel") or not (repo / p).is_file():
            continue
        if "block_tree_cache" not in name:
            continue
        text = (repo / p).read_text()
        for rule in re.finditer(r'^\s*name\s*=\s*"([^"]+)"', text, re.M):
            print(f"//{p.parent}:{rule[1]}")


FIXED_TARGET = "//tools/btc_offline:replay"
REQUIRED_CHECKS = {"//tools/btc_offline:" + name for name in
                   ("metadata_pool_test", "metadata_replay_test", "evaluation_replay_test")}


def validate_fixed_recipe(recipe):
    if recipe.get("target") != FIXED_TARGET:
        raise ValueError(f"source marker must select {FIXED_TARGET}")
    missing = REQUIRED_CHECKS - set(recipe.get("checks", []))
    if missing:
        raise ValueError("missing native checks: " + ", ".join(sorted(missing)))
    contract = recipe.get("dependency_contract", {})
    for field in ("require", "forbid"):
        entries = contract.get(field)
        if not isinstance(entries, list) or not entries or not all(isinstance(x, str) for x in entries):
            raise ValueError(f"source marker must declare dependency_contract.{field}")


def check_graph(output, contract):
    labels = {m[1] for m in re.finditer(r"^\S+ rule (\S+)", output, re.M)}
    missing = set(contract["require"]) - labels
    forbidden = set(contract["forbid"]) & labels
    if missing or forbidden:
        raise ValueError(f"offline dependency check failed: missing={sorted(missing)}, forbidden={sorted(forbidden)}")


def auto_build(a):
    """Build and test the fixed native target from an isolated source checkout."""
    import os
    import sys
    if not a.work:
        raise ValueError("auto requires a new --work directory for the isolated checkout")
    repo, work = a.repo.resolve(), a.work.resolve()
    rev = git(repo, "rev-parse", "HEAD").strip()
    work.mkdir(parents=True, exist_ok=False)
    checkout = work / "source"
    subprocess.run(["git", "clone", "--no-checkout", "--local", str(repo), str(checkout)], check=True)
    subprocess.run(["git", "-C", str(checkout), "checkout", "--detach", rev], check=True)
    marker = checkout / "btc-offline-backend.json"
    if not marker.is_file():
        raise ValueError("missing btc-offline-backend.json source marker; use a checkout with the native offline target")
    recipe = json.loads(marker.read_text())
    validate_fixed_recipe(recipe)
    targets = [recipe["target"], *recipe["checks"]]
    if len(targets) < 2 or any(not re.fullmatch(r"//[\w./+-]+:[\w.+-]+", t) for t in targets):
        raise ValueError("adapter must declare concrete replay and verification targets")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", NVIDIA_VISIBLE_DEVICES="void")
    def run(*args):
        return subprocess.check_output([a.bazel, *args], cwd=checkout, env=env, text=True)
    options = a.option or []
    deps = run("cquery", f"deps({targets[0]})", *options, "--output=label_kind")
    (work / "dependencies.txt").write_text(deps)
    check_graph(deps, recipe["dependency_contract"])
    run("build", *targets, *options)
    binaries = []
    for t in targets:
        files = run("cquery", t, *options, "--output=files").splitlines()
        if len(files) != 1:
            raise ValueError(f"expected one executable for {t}")
        binaries.append(checkout / files[0])
    for check in binaries[1:]:
        subprocess.run([str(check)], cwd=work, env=env, check=True)
    subprocess.run([sys.executable, str(Path(__file__).with_name("smoke.py")), str(binaries[0])],
                   cwd=work, env=env, check=True)
    from prepare import digest
    receipt = dict(build_mode="source", revision=rev, recipe=recipe, options=options, binary=str(binaries[0]),
                   sha256=digest(binaries[0]), checks_passed=targets[1:])
    (work / "build-receipt.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))


def main(a):
    if a.action == "auto":
        return auto_build(a)
    repo = a.repo.resolve()
    rev = git(repo, "rev-parse", "HEAD").strip()
    if a.action == "locate":
        print(rev)
        discover(repo)
        return
    if not a.target or not a.target.startswith("//") or ":" not in a.target or any(
        c in a.target for c in ("...", "*", "(", ")", " ", "\n")
    ) or a.target.endswith((":all", ":all-targets")):
        raise ValueError("select one explicit native replay binary target")
    options = a.option or []
    def bazel(command, expression, extra=()):
        return subprocess.check_output([a.bazel, command, expression, *options, *extra], cwd=repo, text=True)
    # Bazel resolves macros, select() and transitive dependencies using this checkout and build configuration.
    labels = bazel("cquery", f"deps({a.target})", ["--output=label_kind"]).splitlines()
    bazel("build", a.target)
    outputs = bazel("cquery", a.target, ["--output=files"]).splitlines()
    print(json.dumps(dict(revision=rev, target=a.target, options=options, outputs=outputs,
                          worktree_status=git(repo, "status", "--porcelain").splitlines(),
                          dependencies=labels), indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=["locate", "build", "auto"])
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--target")
    p.add_argument("--work", type=Path)
    p.add_argument("--bazel", default="bazelisk")
    p.add_argument("--option", action="append")
    main(p.parse_args())
