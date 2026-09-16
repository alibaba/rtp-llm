"""Merge worker reports without hiding missing or incomplete test execution."""

import argparse
from pathlib import Path
import xml.etree.ElementTree as ET


def merge_reports(paths, output, *, required=(), forbid_skips=False):
    required = {str(path) for path in required}
    paths = [str(path) for path in paths]
    root = ET.Element("testsuites")
    problems = []
    total = 0
    for path in paths:
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
    if total == 0:
        problems.append("remote session executed zero testcases")
    if problems:
        suite = ET.SubElement(root, "testsuite", name="report-integrity", tests="1", errors="1")
        case = ET.SubElement(suite, "testcase", name="complete_execution_reports")
        ET.SubElement(case, "error", message="incomplete remote execution").text = "\n".join(problems)
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
    merge_reports(args.paths, args.output, required=args.required, forbid_skips=args.forbid_skips)


if __name__ == "__main__":
    main()
