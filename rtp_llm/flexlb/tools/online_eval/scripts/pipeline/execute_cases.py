#!/usr/bin/env python3
"""Parallel execution of YAML-configured Python cases with owned port windows."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from adapters import port_windows
from scenario.selection_cli import add_selection_arguments, resolve_selection

def main() -> int:
    parser = argparse.ArgumentParser(
        description="FlexLB case-test parallel orchestrator (lane parallelism)"
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=4,
        help=(
            "lane count (default 4; 1 = single-lane serial run; cap "
            "derived from --mock-stride — 21 at the default 500, 6 at "
            "2000)"
        ),
    )
    parser.add_argument("--retain-tsdb", action="store_true", help="keep stopped TSDB after verified experiment archiving")
    parser.add_argument("--scheduling", choices=("dynamic", "static"), default="dynamic",
                        help="dynamic: idle lanes claim the next run; static: fixed lane lists")
    add_selection_arguments(parser, allow_all_profiles=False)
    parser.add_argument(
        "--grade",
        default="normal",
        choices=("strict", "normal", "loose"),
        help="passed through to the runner",
    )
    parser.add_argument(
        "--json",
        default=None,
        help="aggregated JSON path (default <out-dir>/aggregate.json)",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="lane artifacts dir (default /tmp/flexlb_ft_parallel_<ts>)",
    )
    parser.add_argument(
        "--archive", default=None,
        help="optional single experiment ZIP; includes run plan, results and bounded evidence",
    )
    parser.add_argument(
        "--categories",
        default=None,
        help=(
            "comma-separated category subset to orchestrate (default: all "
            "nine; CLI kebab-case, e.g. engine-fault is accepted as-is)"
        ),
    )
    parser.add_argument(
        "--shard",
        choices=["case"],
        default="case",
        help="partition by compiled instance",
    )
    parser.add_argument(
        "--timing-json",
        default=None,
        help=(
            "case-mode cost baseline: a prior run's aggregate JSON "
            "(cases[].duration_ms). Overrides the auto-maintained shared "
            "baseline (default /tmp/flexlb_ft_timing_baseline.json, env "
            "FLEXLB_FT_TIMING_BASELINE). Missing file → uniform split; "
            "cases absent from the baseline → family-weight fallback "
            "(both warn on stderr)"
        ),
    )
    parser.add_argument(
        "--mock-stride",
        type=int,
        default=None,
        help=(
            "mock-port stride between lanes (default 500). VERIFIED "
            "per-lane mock window: [base-1 .. base+151] = 153 ports "
            "(EnvManager._pick_base_grpc_port / start_victim; JavaMockEngine-"
            "Cluster http=base-1, engines=base..base+n-1, victim zone "
            "base+149..151), so 500 keeps ~3x headroom and a lane cap "
            "of 21 (the dev container sustains 4-8)"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print the lane plan + port matrix and exit (no execution)",
    )
    parser.add_argument(
        "--source",
        choices=["yaml"],
        default="yaml",
        help="instance source (default yaml: Python programs configured by YAML)",
    )
    parser.add_argument(
        "--case-dir",
        default=None,
        help="configuration file/directory (default: bundled scenarios/) for Python case configurations",
    )
    parser.add_argument(
        "--instances",
        default=None,
        help="exact compiled instance IDs for compiled Python cases",
    )
    parser.add_argument("--experiment-params", help="JSON: case -> dotted YAML field -> scalar value")
    args = parser.parse_args()
    resolve_selection(args, parser)
    if args.case_dir is None:
        args.case_dir = str(Path(__file__).resolve().parents[2] / "config/scenarios")
    elif not args.case_dir:
        parser.error("--case-dir must not be empty")

    # Mock stride: CLI --mock-stride over the 500 default.  The verified
    # per-lane mock footprint is [base-1 .. base+151] (153 ports), so any
    # stride >= 153 keeps lanes disjoint; reject below that outright.
    mock_stride = args.mock_stride if args.mock_stride is not None else port_windows.MOCK_PORT_STRIDE
    args.mock_stride = mock_stride
    if mock_stride <= port_windows.MOCK_PORT_WINDOW_LAST + 1:
        parser.error(
            f"--mock-stride must be >= {port_windows.MOCK_PORT_WINDOW_LAST + 2} "
            f"(verified per-lane mock window = {port_windows.MOCK_PORT_WINDOW_LAST + 2} "
            "ports: [base-1 .. base+151])"
        )

    # Lane cap is DERIVED from the stride (not the old fixed 6): the last
    # lane's mock window must stay under 65535; master groups (stride 10)
    # only bind for the truly absurd bases.
    mbase = port_windows.mock_base()
    mabase = port_windows.master_base()
    cap = min(
        port_windows.max_lanes(mock_stride, mbase),
        (65535 - 5 - mabase) // port_windows.MASTER_PORT_STRIDE + 1,
    )
    if not 1 <= args.parallel <= cap:
        parser.error(
            f"--parallel must be 1..{cap} (mock stride {mock_stride}, "
            f"mock base {mbase}, master base {mabase})"
        )

    from scenario.instance_runner import run_structured

    return run_structured(args)


if __name__ == "__main__":
    sys.exit(main())
