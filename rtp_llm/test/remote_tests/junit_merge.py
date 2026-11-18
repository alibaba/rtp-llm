"""Merge worker reports without hiding missing or incomplete test execution."""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

_REPORT_CASE_KIND = "rtp_report_case_kind"
_INTEGRITY_ERROR_KIND = "synthetic-integrity-error"


def count_report_integrity_errors(root):
    """Count only explicitly marked error records emitted by this merger.

    Physical JUnit still contains these errors and must remain failed. They are
    diagnostics about the reports, not an additional executed pytest instance.
    Unmarked legacy records and real nodeids are deliberately not reclassified.
    """
    count = 0
    for suite in root.iter("testsuite"):
        if suite.get("name") != "report-integrity":
            continue
        for case in suite.findall("testcase"):
            properties = case.findall("./properties/property")
            kinds = [
                prop.get("value")
                for prop in properties
                if prop.get("name") == _REPORT_CASE_KIND
            ]
            has_nodeid = any(prop.get("name") == "nodeid" for prop in properties)
            errors = case.findall("error")
            if (
                kinds == [_INTEGRITY_ERROR_KIND]
                and not has_nodeid
                and case.get("name") == "complete_execution_reports"
                and len(errors) == 1
                and errors[0].get("message") == "incomplete remote execution"
                and case.find("failure") is None
                and case.find("skipped") is None
            ):
                count += 1
    return count


def remove_replayed_zero_execution_error(root, cached_cases):
    """Reconcile empty entries only against their complete worker selection proof.

    Return whether every selected instance is accounted for by fresh execution
    or validated cache replay, including mixed sessions.
    """
    evidence = root.find("./properties/property[@name='rtp_cache_collection']")
    if evidence is None or count_report_integrity_errors(root) != 1:
        return False
    try:
        reports = json.loads(evidence.get("value"))
        cached = {
            case.find("./properties/property[@name='nodeid']").get("value")
            for case in cached_cases
        }
        fresh = {
            prop.get("value")
            for prop in root.findall(".//testcase/properties/property[@name='nodeid']")
        }
        available = cached | fresh
        selected = set()
        complete = bool(reports)
        covered_empty_entries = set()
        for report in reports:
            workers = report["workers"]
            if not workers:
                complete = False
                continue
            expected = set(workers[0]["selected"])
            claimed = set(workers[0]["cached"])
            valid = (
                all(
                    set(worker["selected"]) == expected
                    and set(worker["cached"]) == claimed
                    for worker in workers
                )
                and claimed <= expected
                and claimed <= cached
            )
            valid = valid and (expected - claimed) <= fresh
            if report["required"] and not expected:
                valid = False
            complete = complete and valid
            selected.update(expected)
            if valid and expected and claimed == expected:
                covered_empty_entries.add(
                    report["path"] + ": required test entry collected zero cases"
                )
        complete = complete and bool(selected) and selected == available
    except (ValueError, KeyError, TypeError, AttributeError, IndexError):
        return False
    for suite in root.findall("testsuite"):
        if suite.get("name") != "report-integrity":
            continue
        error = suite.find("testcase/error")
        problems = (error.text or "").splitlines() if error is not None else []
        remaining = [
            problem
            for problem in problems
            if problem not in covered_empty_entries
            and not (complete and problem == "remote session executed zero testcases")
        ]
        if remaining:
            error.text = "\n".join(remaining)
            return False
        if problems:
            root.remove(suite)
            return complete
    return False


def merge_reports(paths, output, *, required=(), forbid_skips=False):
    required = {str(path) for path in required}
    paths = [str(path) for path in paths]
    root = ET.Element("testsuites")
    problems = []
    collection_reports = []
    total = 0
    for path in paths:
        workers = []
        for evidence in sorted(
            Path(path).parent.glob(Path(path).name + ".collection.*.json")
        ):
            try:
                workers.append(json.loads(evidence.read_text()))
            except (OSError, ValueError):
                workers.append({})
        collection_reports.append(
            {"path": path, "required": path in required, "workers": workers}
        )
        try:
            report = ET.parse(path).getroot()
            if report.tag not in {"testsuite", "testsuites"}:
                raise ValueError(f"unexpected report root: {report.tag}")
            cases = list(report.iter("testcase"))
            total += len(cases)
            if path in required and not cases:
                problems.append(f"{path}: required test entry collected zero cases")
            for case in cases:
                if case.find("failure") is not None or case.find("error") is not None:
                    problems.append(f"{path}: failed testcase {case.get('name')}")
                if forbid_skips and case.find("skipped") is not None:
                    problems.append(f"{path}: skipped testcase {case.get('name')}")
            suites = [report] if report.tag == "testsuite" else list(report)
            root.extend(suites)
        except (OSError, ET.ParseError, ValueError) as exc:
            problems.append(f"{path}: {exc}")
    for path in sorted(required - set(paths)):
        problems.append(f"{path}: required report was not supplied")
    properties = ET.SubElement(root, "properties")
    ET.SubElement(
        properties,
        "property",
        name="rtp_cache_collection",
        value=json.dumps(collection_reports),
    )
    if total == 0:
        problems.append("remote session executed zero testcases")
    if problems:
        suite = ET.SubElement(
            root, "testsuite", name="report-integrity", tests="1", errors="1"
        )
        case = ET.SubElement(suite, "testcase", name="complete_execution_reports")
        properties = ET.SubElement(case, "properties")
        ET.SubElement(
            properties, "property",
            name=_REPORT_CASE_KIND, value=_INTEGRITY_ERROR_KIND,
        )
        ET.SubElement(case, "error", message="incomplete remote execution").text = (
            "\n".join(problems)
        )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(output, encoding="unicode", xml_declaration=True)
    if problems:
        raise ValueError("\n".join(problems))
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--required", action="append", default=[])
    parser.add_argument("--forbid-skips", action="store_true")
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()
    merge_reports(
        args.paths, args.output, required=args.required, forbid_skips=args.forbid_skips
    )


if __name__ == "__main__":
    main()
