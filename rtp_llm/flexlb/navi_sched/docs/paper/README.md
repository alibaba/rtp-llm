# NaviSched Paper

This directory contains the arXiv-style NaviSched paper and the script that
turns the canonical simulator report into LaTeX tables and plots.

## Regenerate experiment artifacts

From the repository root:

```bash
python3 aios/navi_sched/docs/paper/generate_artifacts.py
```

The script reads
`aios/navi_sched/test/data/cost_scheduler/comparison_report.json` and writes
generated LaTeX fragments under `aios/navi_sched/docs/paper/generated/`.
Override the report when validating an unfinished local run:

```bash
python3 aios/navi_sched/docs/paper/generate_artifacts.py \
  --report aios/navi_sched/tmp/cost_scheduler_debug/comparison_report.json \
  --allow-partial
```

The canonical report itself is produced by:

```bash
aios/navi_sched/scripts/run_cost_scheduler_simulation_suite.sh
```

## Build the paper

The source uses a standard two-column `article` layout and common TeX Live
packages. Build with:

```bash
cd aios/navi_sched/docs/paper
make
```

Run BibTeX manually when `latexmk` is unavailable:

```bash
pdflatex NaviSched-State-Aware-Cost-Optimization-for-Distributed-LLM-Prefill-Scheduling.tex
bibtex NaviSched-State-Aware-Cost-Optimization-for-Distributed-LLM-Prefill-Scheduling
pdflatex NaviSched-State-Aware-Cost-Optimization-for-Distributed-LLM-Prefill-Scheduling.tex
pdflatex NaviSched-State-Aware-Cost-Optimization-for-Distributed-LLM-Prefill-Scheduling.tex
```

The Makefile also falls back to `tectonic`. A user-local environment can be
created without changing the repository toolchain:

```bash
CONDA_SOLVER=classic /opt/conda310/bin/conda create -y \
  -p ~/.conda/envs/navi-paper -c conda-forge tectonic
PATH="$HOME/.conda/envs/navi-paper/bin:$PATH" make
```

For an unfinished same-revision suite, pass the partial report through the
Makefile so `artifacts` does not regenerate from the packaged snapshot:

```bash
PATH="$HOME/.conda/envs/navi-paper/bin:$PATH" make \
  ARTIFACT_ARGS="--report ../../tmp/cost_scheduler_debug/comparison_report.json --allow-partial"
```

`NaviSched-State-Aware-Cost-Optimization-for-Distributed-LLM-Prefill-Scheduling.pdf`
is intentionally not checked in unless it is generated from the same report
revision as the committed `generated/` files.
