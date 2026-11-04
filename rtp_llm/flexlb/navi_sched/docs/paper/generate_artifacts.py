#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_REPORT = (
    ROOT / "rtp_llm/flexlb/navi_sched/test/data/cost_scheduler/"
    "comparison_report.json"
)
DEFAULT_ABLATION = Path(__file__).parent / "data/objective_ablation.json"
DEFAULT_OUTPUT = Path(__file__).parent / "generated"
EXPECTED_GROUP_COUNTS = {
    "Strategy baseline": 3,
    "Single-biz lambda sweep": 18,
    "Batch window sweep": 4,
    "Independent multi-biz lambda sweep": 10,
    "Unified mixed-pool lambda sweep": 10,
    "Independent and unified strategy baseline": 6,
    "Unified pool capacity validation": 2,
    "Small-lambda loop cap sweep": 5,
}


def tex(value):
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "_": r"\_",
        "%": r"\%",
        "&": r"\&",
        "#": r"\#",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def number(value, digits=2):
    if value is None:
        return "--"
    return f"{float(value):.{digits}f}"


def lambda_number(value):
    if value is None:
        return "--"
    numeric = float(value)
    if numeric == 0.0 or numeric == 1.0:
        return f"{numeric:.1f}"
    return f"{numeric:.5f}".rstrip("0")


def percent(value, digits=1):
    if value is None:
        return "--"
    return f"{100.0 * float(value):.{digits}f}"


def compact_number(value):
    if value is None:
        return "--"
    return f"{float(value):.5g}"


def average(run, metric):
    value = run.get(metric) or {}
    return value.get("average_ms")


def p99(run, metric):
    value = run.get(metric) or {}
    return value.get("p99_ms")


def schedule_latency_ms(run):
    value = (run.get("schedule_latency_us") or {}).get("average")
    return None if value is None else float(value) / 1000.0


def group_runs(runs, name):
    return [run for run in runs if run.get("group") == name]


def find_run(runs, group, label):
    return next(
        run for run in runs
        if run.get("group") == group and run.get("label") == label
    )


def prediction_total(run):
    return (run.get("decision_prediction_accuracy") or {}).get("total") or {}


def prediction_sample_count(run):
    return (run.get("decision_prediction_accuracy") or {}).get("sample_count")


def validate_report(report, allow_partial):
    runs = report.get("runs", [])
    if not runs:
        raise ValueError("report has no runs")
    if report.get("experiment", {}).get("time_compression") != 1.0:
        raise ValueError("paper results require time compression 1")
    for run in runs:
        if run.get("agent_readiness") != "all_agents_ready":
            raise ValueError(
                f"run is not all-agents-ready: {run.get('label')}"
            )
    if allow_partial:
        return
    for group, expected_count in EXPECTED_GROUP_COUNTS.items():
        actual_count = len(group_runs(runs, group))
        if actual_count != expected_count:
            raise ValueError(
                f"group {group!r} has {actual_count} runs; "
                f"expected {expected_count}"
            )


def table_header(caption, label, columns, headers):
    return [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{2.4pt}",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{columns}}}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]


def table_footer(note):
    return [
        r"\bottomrule",
        r"\end{tabular}",
        r"\par\vspace{3pt}",
        r"\begin{minipage}{0.98\textwidth}",
        r"\footnotesize " + note,
        r"\end{minipage}",
        r"\end{table*}",
        "",
    ]


def run_row(run, include_lambda=True):
    row = [tex(run.get("label", run.get("scheduler", "")))]
    if include_lambda:
        row.append(lambda_number(run.get("lambda")))
    loops = (run.get("sampled_loop_count") or {}).get("average")
    row.extend([
        number(loops, 1),
        number(run.get("average_requests_per_schedule"), 1),
        number(schedule_latency_ms(run), 3),
        number(average(run, "queue_wait"), 2),
        number(average(run, "execution_latency"), 2),
        number(average(run, "end_to_end_latency"), 2),
        number(p99(run, "end_to_end_latency"), 2),
        percent(run.get("cache_hit_rate"), 1),
        percent(run.get("average_utilization"), 1),
        number(run.get("completion_throughput_qps"), 1),
    ])
    return " & ".join(row) + r" \\"


def make_strategy_table(runs):
    selected = group_runs(runs, "Strategy baseline")
    cost = find_run(runs, "Strategy baseline", "cost")
    lines = table_header(
        "Single-workload strategy comparison. Utilization is logical "
        "busy time divided by logical elapsed time.",
        "tab:strategy-baseline",
        "lrrrrrrrrrrr",
        [
            "Scheduler", "$\\lambda$", "Loops", "Req./sched.",
            "Sched. ms", "Queue", "Exec.", "E2E", "P99", "Cache",
            "Util.", "QPS",
        ],
    )
    lines.extend(run_row(run) for run in selected)
    lines.extend(table_footer(
        "The table compares scheduler overhead and service outcomes under "
        "the same single-workload trace to test whether joint cost-aware "
        "routing avoids overload. CostScheduler preserves the offered "
        f"{number(cost.get('completion_throughput_qps'), 1)} QPS with low "
        "queueing, whereas random and cache-only routing "
        "build long queues despite lower overhead or higher cache hit rate."
    ))
    return lines


def make_lambda_table(runs, group, caption, label, note):
    selected = group_runs(runs, group)
    lines = table_header(
        caption,
        label,
        "lrrrrrrrrrrr",
        [
            "Run", "$\\lambda$", "Loops", "Req./sched.", "Sched. ms",
            "Queue", "Exec.", "E2E", "P99", "Cache", "Util.", "QPS",
        ],
    )
    lines.extend(run_row(run) for run in selected)
    lines.extend(table_footer(note))
    return lines


def make_pool_table(runs):
    selected = group_runs(
        runs, "Independent and unified strategy baseline"
    )
    independent = find_run(
        runs, "Independent and unified strategy baseline", "independent cost"
    )
    unified = find_run(
        runs, "Independent and unified strategy baseline", "unified cost"
    )
    lines = table_header(
        "Independent pools and one unified mixed-workload pool at equal "
        "aggregate capacity.",
        "tab:pool-baseline",
        "lrrrrrrrrrrr",
        [
            "Deployment", "$\\lambda$", "Loops", "Req./sched.",
            "Sched. ms", "Queue", "Exec.", "E2E", "P99", "Cache",
            "Util.", "QPS",
        ],
    )
    lines.extend(run_row(run) for run in selected)
    lines.extend(table_footer(
        "The table holds aggregate capacity fixed while comparing policies "
        "and independent versus unified placement, isolating the benefit "
        "of shared scheduling. The unified CostScheduler run changes mean "
        f"E2E from {number(average(independent, 'end_to_end_latency'), 2)} "
        f"ms to {number(average(unified, 'end_to_end_latency'), 2)} ms "
        "relative to independent CostScheduler "
        "without reducing completion throughput."
    ))
    return lines


def make_window_table(runs):
    selected = group_runs(runs, "Batch window sweep")
    first = selected[0]
    last = selected[-1]
    lines = table_header(
        "Effect of the Master collection window.",
        "tab:window-sweep",
        "lrrrrrrrrrrr",
        [
            "Window", "$\\lambda$", "Loops", "Req./sched.",
            "Sched. ms", "Queue", "Exec.", "E2E", "P99", "Cache",
            "Util.", "QPS",
        ],
    )
    lines.extend(run_row(run) for run in selected)
    lines.extend(table_footer(
        "The table varies the request-count and wait-time limits together "
        "to expose the batching-versus-delay tradeoff. From the smallest to "
        "the largest window, mean batch size rises from "
        f"{number(first.get('average_batch_size'), 3)} to "
        f"{number(last.get('average_batch_size'), 3)} and card-time falls, "
        f"but mean E2E rises from {number(average(first, 'end_to_end_latency'), 2)} "
        f"to {number(average(last, 'end_to_end_latency'), 2)} ms and P99 also "
        "increases. The largest window is therefore not uniformly best."
    ))
    return lines


def make_loop_table(runs):
    selected = group_runs(runs, "Small-lambda loop cap sweep")
    cap2 = find_run(runs, "Small-lambda loop cap sweep", "cap 2")
    cap50 = find_run(runs, "Small-lambda loop cap sweep", "cap 50")
    lines = table_header(
        "Optimization-loop cap under the shared mixed workload.",
        "tab:loop-sweep",
        "lrrrrrrrrrrr",
        [
            "Cap", "$\\lambda$", "Loops", "Req./sched.", "Sched. ms",
            "Queue", "Exec.", "E2E", "P99", "Cache", "Util.", "QPS",
        ],
    )
    lines.extend(run_row(run) for run in selected)
    lines.extend(table_footer(
        "The table varies only the optimization-loop cap on the shared "
        "workload to measure route quality against scheduler cost. In this "
        "trace, raising the cap from two to fifty changes mean E2E from "
        f"{number(average(cap2, 'end_to_end_latency'), 2)} to "
        f"{number(average(cap50, 'end_to_end_latency'), 2)} ms and mean "
        f"scheduling time from {number(schedule_latency_ms(cap2), 3)} to "
        f"{number(schedule_latency_ms(cap50), 3)} ms. Deeper optimization "
        "does not improve the service outcome in this run."
    ))
    return lines


def make_prediction_table(runs):
    selected = [
        find_run(runs, "Strategy baseline", "cost"),
        find_run(
            runs,
            "Independent and unified strategy baseline",
            "independent cost",
        ),
        find_run(
            runs,
            "Independent and unified strategy baseline",
            "unified cost",
        ),
        find_run(runs, "Small-lambda loop cap sweep", "cap 2"),
        find_run(runs, "Unified mixed-pool lambda sweep", "l1"),
    ]
    display = [
        "Single, $\\lambda=0.5$",
        "Independent, $\\lambda=0$",
        "Unified, $\\lambda=0$",
        "Unified cap 2, $\\lambda=0$",
        "Unified, $\\lambda=1$",
    ]
    lines = table_header(
        "Decision-time total-latency prediction versus posterior observation.",
        "tab:canonical-prediction",
        "lrrrr",
        ["Case", "Samples", "Pred. ms", "Actual ms", "MAPE"],
    )
    for name, run in zip(display, selected):
        prediction = prediction_total(run)
        row = [
            name,
            str(prediction_sample_count(run) or "--"),
            number(prediction.get("predicted_mean_ms"), 2),
            number(prediction.get("actual_mean_ms"), 2),
            (number(prediction.get("mape_pct"), 2) + r"\%"),
        ]
        lines.append(" & ".join(row) + r" \\")
    lines.extend(table_footer(
        "Prediction is the value available before routing; actual is the "
        "posterior queue-plus-execution observation for the matched batch. "
        "All unmatched-batch counts are zero. The canonical configuration "
        "uses a 100,000-sample fit batch with interval zero, so these runs do "
        "not trigger online fitting and evaluate the default static model. "
        "The high heterogeneous-workload MAPE is therefore a model-accuracy "
        "limitation, not evidence of precise steady-state prediction."
    ))
    return lines


def make_workload_table(runs):
    candidates = [run for run in runs if run.get("traffic_segments")]
    source = next(
        (
            run for run in candidates
            if run.get("group") == "Unified mixed-pool lambda sweep" and
            run.get("lambda") == 0.0
        ),
        None,
    )
    if source is None:
        source = max(
            candidates,
            key=lambda run: len(run.get("traffic_segments", [])),
            default=None,
        )
    lines = table_header(
        "Trace composition and per-source outcomes in the unified "
        "CostScheduler run at lambda zero.",
        "tab:workloads",
        "lrrrrrrrr",
        [
            "Trace", "Requests", "Share", "Mean", "P50", "P90", "P99",
            "Cache", "Batch",
        ],
    )
    if source:
        for segment in source.get("traffic_segments", []):
            prompt = segment.get("prompt_token_distribution") or {}
            cache = segment.get("actual_cache") or {}
            cache_rate = cache.get("hit_rate", cache.get("cache_hit_rate"))
            batch = segment.get("batch_size_distribution") or {}
            row = [
                tex(segment.get("name", "")),
                str(segment.get("request_count", "--")),
                percent(segment.get("request_share"), 1),
                number(prompt.get("average"), 1),
                number(prompt.get("p50"), 0),
                number(prompt.get("p90"), 0),
                number(prompt.get("p99"), 0),
                percent(cache_rate, 1),
                number(batch.get("average"), 2),
            ]
            lines.append(" & ".join(row) + r" \\")
    lines.extend(table_footer(
        "The table characterizes the five source traces mixed into one pool "
        "so aggregate results can be interpreted against workload skew. "
        "Mean prompt length spans 6.5k to 234k tokens, and the longest source "
        "has both the lowest cache hit rate and almost no batching."
    ))
    return lines


def make_ablation_table(ablation):
    lines = table_header(
        "Objective ablation on identical schedule contexts. All loop counts "
        "are measured rather than configured values.",
        "tab:objective-ablation",
        "llrrrrrrr",
        [
            "Weight", "$\\lambda$", "Loops", "Queue", "Exec.",
            "E2E", "P99", "Cache", "Batch",
        ],
    )
    for run in ablation.get("runs", []):
        formula = (
            "linear" if run["formula"] == "linear" else
            r"$\log(1+128x)/128$"
        )
        row = [
            formula,
            number(run.get("lambda"), 1),
            str(run.get("actual_loops", "--")),
            number(run.get("queue_ms"), 2),
            number(run.get("execution_ms"), 2),
            number(run.get("e2e_ms"), 2),
            number(run.get("e2e_p99_ms"), 2),
            percent(run.get("cache_hit_rate"), 1),
            number(run.get("average_batch_size"), 2),
        ]
        lines.append(" & ".join(row) + r" \\")
    lines.extend(table_footer(
        "The table compares linear and logarithmic latency weights on "
        "identical scheduling contexts to test the dispersion correction. "
        "The logarithmic weight prevents queue growth at deeper optimization; "
        "five loops gives the lowest E2E at lambda zero, while further loops "
        "mainly improve the projected objective."
    ))
    return lines


def make_loss_table(runs):
    groups = [
        ("Single-biz lambda sweep", "Single"),
        ("Independent multi-biz lambda sweep", "Independent"),
        ("Unified mixed-pool lambda sweep", "Unified"),
    ]
    lines = table_header(
        "Distribution of one deduplicated CostScheduler objective value per "
        "scheduling round. Losses are comparable only within one lambda and "
        "workload configuration.",
        "tab:loss-distribution",
        "llrrrrrr",
        [
            "Deployment", "$\\lambda$", "Rounds", "Mean", "P50",
            "P90", "P99", "Max",
        ],
    )
    first_group = True
    for group, display in groups:
        if not first_group:
            lines.append(r"\midrule")
        first_group = False
        for run in group_runs(runs, group):
            loss = run.get("loss") or {}
            row = [
                display,
                lambda_number(run.get("lambda")),
                str(loss.get("count", "--")),
                compact_number(loss.get("average")),
                compact_number(loss.get("p50")),
                compact_number(loss.get("p90")),
                compact_number(loss.get("p99")),
                compact_number(loss.get("max")),
            ]
            lines.append(" & ".join(row) + r" \\")
    lines.extend(table_footer(
        "The table reports the distribution of one deduplicated final "
        "objective per scheduling round to expose optimizer spread and "
        "outliers. Values are useful within an identical workload and "
        "lambda, but their decline as lambda changes must not be interpreted "
        "as a service-level improvement."
    ))
    return lines


def plot_coordinates(runs, key):
    values = []
    for run in runs:
        lambda_value = run.get("lambda")
        if lambda_value is None:
            continue
        if key == "e2e":
            value = average(run, "end_to_end_latency")
        elif key == "cache":
            value = 100.0 * run.get("cache_hit_rate", 0.0)
        else:
            value = 100.0 * run.get("average_utilization", 0.0)
        values.append((float(lambda_value), float(value)))
    values.sort()
    return " ".join(f"({x:.5f},{y:.5f})" for x, y in values)


def make_plots(runs):
    groups = [
        ("Single-biz lambda sweep", "Single workload"),
        ("Independent multi-biz lambda sweep", "Independent pools"),
        ("Unified mixed-pool lambda sweep", "Unified pool"),
    ]
    lines = [
        r"\begin{figure*}[t]",
        r"\centering",
        r"\begin{tikzpicture}",
        r"\begin{groupplot}[",
        r"group style={group size=3 by 1, horizontal sep=1.2cm},",
        r"width=0.31\textwidth, height=0.22\textwidth,",
        r"xlabel={$\lambda$}, grid=major, legend style={font=\scriptsize},",
        r"tick label style={font=\scriptsize}, label style={font=\small}]",
    ]
    metrics = [
        ("e2e", "Mean E2E (ms)", "ymode=log"),
        ("cache", "Cache hit (\\%)", ""),
        ("util", "Mean utilization (\\%)", ""),
    ]
    for metric, ylabel, options in metrics:
        suffix = f",{options}" if options else ""
        lines.append(f"\\nextgroupplot[ylabel={{{ylabel}}}{suffix}]")
        for group, display in groups:
            coordinates = plot_coordinates(group_runs(runs, group), metric)
            lines.append(f"\\addplot coordinates {{{coordinates}}};")
            if metric == "e2e":
                lines.append(f"\\addlegendentry{{{display}}}")
    lines.extend([
        r"\end{groupplot}",
        r"\end{tikzpicture}",
        r"\caption{The CostScheduler tradeoff across the sampled "
        r"$\lambda$ range. Lower $\lambda$ emphasizes online latency; "
        r"higher $\lambda$ emphasizes execution efficiency.}",
        r"\label{fig:lambda-sweep}",
        r"\end{figure*}",
        "",
    ])
    return lines


def write(path, lines):
    path.write_text("\n".join(lines).rstrip("\n") + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--ablation", type=Path, default=DEFAULT_ABLATION)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    ablation = json.loads(args.ablation.read_text(encoding="utf-8"))
    validate_report(report, args.allow_partial)
    runs = report.get("runs", [])
    args.output.mkdir(parents=True, exist_ok=True)

    tables = [
        "% Generated by generate_artifacts.py; do not edit.",
        "\\newcommand{\\ExperimentGeneratedAt}{" +
        tex(report.get("generated_at", "unknown")) + "}",
        "",
    ]
    tables.extend(make_strategy_table(runs))
    tables.extend(make_workload_table(runs))
    tables.extend(make_ablation_table(ablation))
    tables.extend(make_lambda_table(
        runs,
        "Single-biz lambda sweep",
        "Single-workload CostScheduler sweep.",
        "tab:single-lambda",
        "The table sweeps lambda on one workload to show how execution "
        "efficiency trades against online queueing. Low lambda gives the "
        "best latency region; values near one sharply increase queues and "
        "eventually reduce throughput, so no single setting dominates all "
        "objectives.",
    ))
    tables.extend(make_pool_table(runs))
    tables.extend(make_prediction_table(runs))
    tables.extend(make_lambda_table(
        runs,
        "Independent multi-biz lambda sweep",
        "CostScheduler sweep with five independent pools.",
        "tab:independent-lambda",
        "The table sweeps lambda while each workload retains its own Agent "
        "pool, providing a control for the shared-pool experiment. Latency "
        "is broadly stable through the middle range, but degrades at 0.99 "
        "and collapses at the queue-blind lambda-one boundary.",
    ))
    tables.extend(make_lambda_table(
        runs,
        "Unified mixed-pool lambda sweep",
        "CostScheduler sweep after combining five workloads and their nodes.",
        "tab:unified-lambda",
        "The table applies the same sweep after all workloads share one Agent "
        "pool, testing whether the online operating region survives mixing. "
        "Low-lambda settings remain close on this trace, while near-one "
        "settings build large queues and lose throughput despite lower "
        "execution time.",
    ))
    tables.extend(make_window_table(runs))
    tables.extend(make_loop_table(runs))
    tables.extend(make_loss_table(runs))
    write(args.output / "experiment_tables.tex", tables)
    write(
        args.output / "lambda_plots.tex",
        ["% Generated by generate_artifacts.py; do not edit."] +
        make_plots(runs),
    )
    print(f"generated artifacts from {args.report}")
    print(f"output: {args.output}")
    print(f"runs: {len(runs)}")


if __name__ == "__main__":
    main()
