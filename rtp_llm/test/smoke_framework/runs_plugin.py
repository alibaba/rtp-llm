"""pytest plugin: --runs-per-test=N — bazel-style test repetition.

After collection (and after `-k` filtering), every collected item is
duplicated N times. Each replica gets a unique nodeid suffix
``[run01/N]`` so REAPI dispatches them as N independent actions in
parallel — fresh server, fresh CUDA context per run. Use for
flakiness / determinism investigations:

    pytest -k "bf16 or beam_search_tp2" --runs-per-test=10 --remote

Replicas are created via `Function.from_parent` so each gets its own
funcargs / fixture lifecycle (a `copy.copy` would share `funcargs=None`
across reps and crash setup with `argument of type 'NoneType'`).
Result reporting uses the suffixed nodeid so PASS/FAIL is per replica.

Notes:
- `--runs-per-test=1` (default) is a no-op.
- Replication happens AFTER `-k`/`-m` filtering, so combine with those
  flags to scope cost.
- Run #1 keeps the ORIGINAL nodeid (for grep tooling that targets the
  single-run baseline); runs #2..N get the `[run##/N]` suffix.
"""

from __future__ import annotations

from typing import List


def pytest_addoption(parser):
    g = parser.getgroup("smoke", "Smoke framework — test repetition")
    g.addoption(
        "--runs-per-test",
        type=int,
        default=1,
        metavar="N",
        help=(
            "Run every collected test N times (bazel --runs_per_test "
            "analogue). Each rep is an independent pytest item — under "
            "--remote each goes to its own REAPI worker. Default: 1."
        ),
    )


def _clone_item(item, new_name: str):
    """Make a fresh sibling Function item with a different name (and thus nodeid).

    `Function.from_parent` invokes the pytest constructor protocol so
    funcargs/fixtureinfo are reinitialized — `copy.copy` shares these
    and breaks setup (funcargs=None on the clone).
    """
    cls = type(item)
    return cls.from_parent(
        parent=item.parent,
        name=new_name,
        callspec=getattr(item, "callspec", None),
        fixtureinfo=getattr(item, "_fixtureinfo", None),
        keywords=dict(item.keywords),
        originalname=getattr(item, "originalname", None),
    )


def pytest_collection_modifyitems(session, config, items: List):
    n = int(config.getoption("--runs-per-test", default=1) or 1)
    if n <= 1:
        return
    width = max(2, len(str(n)))
    repeated = []
    for it in items:
        for i in range(1, n + 1):
            if i == 1:
                # First rep keeps original name+nodeid (single-run baseline).
                repeated.append(it)
                continue
            new_name = "{}[run{:0{w}d}/{}]".format(it.name, i, n, w=width)
            repeated.append(_clone_item(it, new_name))
    items[:] = repeated
