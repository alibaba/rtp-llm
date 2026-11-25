"""Deliver mode-aware AOne CI configuration without importing the runtime."""

import argparse
import json
import subprocess
from pathlib import Path


def deliver_config(mode: str, spec: str, checkout: Path, destination: Path) -> dict:
    """Select the CI config from explicit arguments and write delivered JSON.

    Model, topology and resource settings come exclusively from the selected
    JSON. Empty or auto selects the mode-specific default example.
    """
    mode = mode.strip() or "decode"
    if mode not in ("decode", "prefill"):
        raise ValueError("--mode must be 'decode' or 'prefill'")
    spec = spec.strip()
    source_dir = checkout.resolve()
    if spec in ("", "auto"):
        relative = Path("rtp_llm/test/perf_test/examples") / (
            "prefill_aone_ci.json" if mode == "prefill" else "decode_aone_gang.json"
        )
        public_checkout = (
            checkout / "github-opensource"
            if (checkout / "github-opensource").is_dir()
            else checkout
        )
        source_path = (public_checkout / relative).resolve()
        source_dir = source_path.parent
        source = source_path.read_text(encoding="utf-8")
    else:
        if spec.startswith("{"):
            source = spec
        else:
            source_path = (checkout / spec).resolve()
            source_dir = source_path.parent
            source = source_path.read_text(encoding="utf-8")
    base = json.loads(source)
    if not isinstance(base, dict):
        raise ValueError("config must be a JSON object")
    base.setdefault("execution", {})["backend"] = "aone"

    # The public entry materializes these files before packaging remote inputs.
    # Preserve their original base directory when relocating the source JSON.
    def rebase_distributions(value):
        if isinstance(value, dict):
            for key, child in value.items():
                if key == "distribution_file" and isinstance(child, str):
                    path = Path(child)
                    if not path.is_absolute():
                        value[key] = str((source_dir / path).resolve())
                else:
                    rebase_distributions(child)
        elif isinstance(value, list):
            for child in value:
                rebase_distributions(child)

    rebase_distributions(base.get("workload", {}))
    # Both mode gates and the always-run upload consume this artifact root.
    base.setdefault("output", {})["result_root"] = "perf_test_results"
    destination.write_text(json.dumps(base, indent=2) + "\n", encoding="utf-8")
    return base


def record_repository_revisions(checkout: Path) -> dict:
    """Record available checkouts; the paired internal parent is optional."""
    checkout = checkout.resolve()
    repositories = [("public", checkout)]
    if (checkout.parent / "internal_source").is_dir():
        repositories.append(("internal", checkout.parent))
    revisions = {
        name: subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
        for name, repo in repositories
    }
    from rtp_llm.test.perf_test.run_context import write_run_context

    context_path = checkout / "perf_test_results/run_context.json"
    context_path.parent.mkdir(parents=True, exist_ok=True)
    write_run_context(context_path, repository_revisions=revisions)
    return revisions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, default=Path.cwd())
    parser.add_argument(
        "--destination", type=Path, default=Path("multi_node_perf_config.json")
    )
    parser.add_argument("--record-provenance", action="store_true")
    parser.add_argument("--mode", choices=("prefill", "decode"), default="decode")
    parser.add_argument("--config", default="auto")
    args = parser.parse_args()
    if args.record_provenance:
        print(json.dumps(record_repository_revisions(args.checkout), sort_keys=True))
        return
    print(
        json.dumps(
            deliver_config(args.mode, args.config, args.checkout, args.destination),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
