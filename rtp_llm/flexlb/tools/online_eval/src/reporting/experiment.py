"""Frozen view reconstruction and publication over checked experiment containers."""

import json
import re
from pathlib import Path

from artifacts.archive import _secret_like
from artifacts.experiment_store import ArtifactEntry as ArtifactEntry
from artifacts.experiment_store import ArtifactRole as ArtifactRole
from artifacts.experiment_store import ExperimentReader as ContainerReader
from artifacts.experiment_store import ExperimentWriter as ContainerWriter
from artifacts.experiment_store import ViewDescriptor as ViewDescriptor
from schema_contract import matches_schema


class ExperimentWriter(ContainerWriter):
    def add_view(self, name, spec, declaration=None):
        """Keep layout separate from full point arrays; repeated curves share blobs."""
        from reporting.spec import validate

        validate(spec, versioned=True)
        projected = dict(spec)
        panels = []
        for panel_index, panel in enumerate(spec.get("panels", [])):
            panel = dict(panel)
            curves = []
            for curve_index, curve in enumerate(panel.get("series", [])):
                curve = dict(curve)
                for field in ("points", "statistics_points"):
                    if curve.get(field) is not None:
                        path = f"views/{name}/datasets/{panel_index}-{curve_index}-{field}.json"
                        self.add_json(path, curve[field], ArtifactRole.VIEW)
                        curve[field] = {"dataset": path}
                curves.append(curve)
            panel["series"] = curves
            panels.append(panel)
        projected["panels"] = panels
        path = self.add_json(f"views/{name}/layout.json", projected, ArtifactRole.VIEW)
        self.views[name] = ViewDescriptor(path, declaration)


class ExperimentReader(ContainerReader):
    def view_spec(self, name):
        spec = self.read_json(self.views[name].layout)
        for panel in spec.get("panels", []):
            for curve in panel.get("series", []):
                for field in ("points", "statistics_points"):
                    if curve.get(field) is not None:
                        reference = curve[field]
                        if not isinstance(reference, dict) or set(reference) != {"dataset"}:
                            raise ValueError("invalid view dataset reference")
                        if self.entries[reference["dataset"]].role is not ArtifactRole.VIEW:
                            raise ValueError("invalid view dataset role")
                        curve[field] = self.read_json(reference["dataset"])
        from reporting.spec import validate

        validate(spec, versioned=True)
        return spec

    def render(self, name, output):
        from file_io import write_text
        from reporting import render

        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        if name in self.views:
            spec = self.view_spec(name)
        elif name == "default.yaml" and "analysis.json" in self.entries:
            from reporting.workload import build_spec

            definition = self.read_json("experiment.json")
            from reporting.run_model import RunPresentation
            from reporting.view_model import ViewDefinition

            analysis = self.analysis()

            def read_gate(name):
                with self.open("run/" + name) as stream:
                    return stream.read()

            run = RunPresentation.read(analysis, read_gate=read_gate)
            spec = build_spec(analysis, None, presentation=ViewDefinition.read(definition["views"][name]), run=run)
        else:
            raise ValueError("view was not frozen in this experiment: " + name)
        write_text(output, render(spec))
        return output

    @property
    def available_views(self):
        names = list(self.views)
        if "analysis.json" in self.entries and "default.yaml" not in names:
            names.append("default.yaml")
        return names

    @property
    def preferred_view(self):
        definition = self.read_json("experiment.json")
        names = self.available_views
        for name in definition["plan"].get("reports", []):
            if name != "default.yaml" and name in names:
                return name
        if names:
            return names[0]
        raise ValueError("this experiment has no renderable view")

    def analysis(self):
        document = self.read_json("analysis.json")
        if not any(matches_schema(document, "experiment_analysis_schema_version", version) for version in (1, 2)):
            raise ValueError("invalid experiment analysis")
        from monitoring.metric_store import MetricStore

        store = MetricStore(self.read_json(document["metrics"]))
        result = {}
        if document["experiment_analysis_schema_version"] == 2:
            if document.get("execution") != "execution.json":
                raise ValueError("analysis requires canonical execution")
            execution = self.read_json("execution.json")
            fields = document.get("execution_fields")
            if (
                not isinstance(fields, list)
                or any(type(key) is not str or key not in execution for key in fields)
                or len(set(fields)) != len(fields)
            ):
                raise ValueError("invalid analysis execution fields")
            result = {key: execution[key] for key in fields}
        result.update(document["metadata"])
        result["series"], result["statistic_sources"], _, _ = store.series(document["anchor_epoch_s"])
        return result


def _role(path):
    name = path.name
    if name == "metrics.json" or name == "queries.json":
        return ArtifactRole.OBSERVATIONS
    if name.endswith("-gate-evidence.json"):
        return ArtifactRole.GATE_INPUT
    if name.endswith("-gate-result.json"):
        return ArtifactRole.GATE_RESULT
    if name.endswith("samples.jsonl"):
        return ArtifactRole.OBSERVATIONS
    if name in {"client_events.jsonl", "client_lifecycle.jsonl"}:
        return ArtifactRole.REQUESTS
    if name in {
        "flow-input.json",
        "environment.json",
        "actual-master-config.json",
        "master_config.json",
        "master-artifact.json",
        "perf.json",
    }:
        return ArtifactRole.CONFIGURATION
    if name == "analysis.json":
        return ArtifactRole.ANALYSIS
    return ArtifactRole.EXTENSION


def publish_run(directory, definition, execution, *, analysis=None, deadline=None):
    """Package settled per-run evidence. HTML is a regenerable cache, not input."""
    from reporting.core import read_bundle

    directory = Path(directory)
    output, raw_output = directory / "experiment.eval.zip", directory / "experiment.raw.zip"
    definition_path = directory / "experiment-definition.json"
    files = sorted(directory.rglob("*"))
    primary = ExperimentWriter(output, deadline=deadline)
    with primary.publication():
        raw = ExperimentWriter(primary.scratch / "raw.zip", kind="raw", deadline=deadline)
        with raw.publication():
            primary.add_json("experiment.json", definition, ArtifactRole.DEFINITION)
            primary.add_json("execution.json", execution, ArtifactRole.EXECUTION)
            for path in files:
                primary.check()
                if path.is_symlink():
                    primary.omitted.append(dict(path=str(path.relative_to(directory)), reason="symbolic link"))
                    continue
                if not path.is_file() or path in {output, raw_output, definition_path, directory / "result.json"}:
                    continue
                if path.parent == directory and re.fullmatch(r"experiment-[0-9a-f]{64}\.raw\.zip", path.name):
                    continue
                relative = path.relative_to(directory)
                if _secret_like(relative):
                    primary.omitted.append(dict(path=str(relative), reason="secret-like filename"))
                    continue
                # Report manifests belong to the materialized directory protocol;
                # this container has its own checked index and frozen view layouts.
                if relative.parts[0] == "reports":
                    continue  # Views are projections of the canonical facts below.
                if path.suffix.lower() == ".html":
                    continue
                structured = path.suffix.lower() in {".json", ".jsonl", ".yaml", ".yml", ".csv"}
                evidence_archive = path.name.startswith("series-") and path.name.endswith(".json.gz")
                writer = primary if structured or evidence_archive else raw
                writer.add_file(
                    "run/" + relative.as_posix(),
                    path,
                    _role(path) if structured else ArtifactRole.OBSERVATIONS if evidence_archive else ArtifactRole.RAW,
                )
            metrics = directory / "metrics.json"
            if metrics.is_file():
                from monitoring.metric_store import MetricStore

                MetricStore.read(directory)  # Validate the canonical definitions and observations.
                if analysis is not None:
                    shared = [
                        key
                        for key, value in analysis.items()
                        if key not in {"series", "statistic_sources"}
                        and key in execution
                        and json.dumps(value, sort_keys=True, allow_nan=False)
                        == json.dumps(execution[key], sort_keys=True, allow_nan=False)
                    ]
                    primary.add_json(
                        "analysis.json",
                        dict(
                            experiment_analysis_schema_version=2,
                            metadata={
                                key: value
                                for key, value in analysis.items()
                                if key not in {"series", "statistic_sources"} and key not in shared
                            },
                            execution="execution.json",
                            execution_fields=shared,
                            metrics="run/metrics.json",
                            anchor_epoch_s=analysis["clock_anchor"]["epoch_s"],
                        ),
                        ArtifactRole.ANALYSIS,
                    )
            reports = execution.get("workload", {}).get("reports", {})
            view_names = {str(Path(path).parent.resolve()): name for name, path in reports.items()}
            invalid_views = []
            for manifest in sorted((directory / "reports/run").glob("*/manifest.json")):
                try:
                    bundle = read_bundle(manifest.parent)
                    with (bundle / "report-spec.json").open(encoding="utf-8") as stream:
                        spec = json.load(stream)
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    invalid_views.append(dict(view=manifest.parent.name, error=str(exc)))
                    continue
                name = view_names.get(str(bundle.resolve()), bundle.name)
                primary.add_view(name, spec, definition.get("views", {}).get(name))
            missing = [
                name
                for name in ("metrics.json", "workload-evidence.json")
                if execution.get("test_kind") == "workload" and not (directory / name).is_file()
            ]
            missing.extend(
                name.replace("-result.json", "-manifest.json")
                for name in primary.entries
                if name.endswith("-gate-result.json") and name.replace("-result.json", "-manifest.json") not in primary.entries
            )
            primary.add_json(
                "completeness.json",
                dict(
                    missing=missing,
                    omissions=primary.omitted,
                    outcome=execution.get("outcome"),
                    invalid_views=invalid_views,
                    available_views=list(primary.views),
                ),
                ArtifactRole.EXECUTION,
            )
        from file_digest import sha256_file

        digest = sha256_file(raw.output)
        raw_output = directory / ("experiment-" + digest + ".raw.zip")
        # Immutable attachment names preserve an earlier committed primary when
        # republishing fails. A primary never points to an overwritten raw ZIP.
        if raw_output.exists():
            if sha256_file(raw_output) != digest:
                raise ValueError("raw attachment checksum mismatch")
        else:
            raw.output.replace(raw_output)
        primary.add_json(
            "raw-attachment.json",
            dict(
                file=raw_output.name,
                sha256=digest,
                size=raw_output.stat().st_size,
                files={name: entry.to_dict() for name, entry in raw.entries.items()},
            ),
            ArtifactRole.EXECUTION,
        )
    return dict(experiment=str(output), raw=str(raw_output))


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Inspect, verify or render a single experiment file")
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--view", help="frozen view name from the artifact index")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if args.view and not args.out:
        parser.error("--view requires --out")
    with ExperimentReader(args.artifact) as reader:
        if args.verify:
            reader.verify()
        if args.out:
            reader.render(args.view or reader.preferred_view, args.out)
        print(
            json.dumps(
                dict(kind=reader.kind, views=reader.available_views, files=len(reader.entries), omitted=reader.omitted), ensure_ascii=False
            )
        )


if __name__ == "__main__":
    main()
