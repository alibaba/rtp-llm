"""Provision a native Grafana dashboard and saved execution annotations in an isolated session."""

import base64
import copy
import json
import shutil
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import yaml

from ..dashboard import validate_dashboard

DATASOURCE = {"type": "prometheus", "uid": "mock-prometheus"}


def prepare_grafana_static_root(directory, grafana_home):
    public = grafana_home / "public"
    static_root = directory / "public"
    views = static_root / "views"
    views.mkdir(parents=True)
    for path in public.iterdir():
        if path.name != "views":
            (static_root / path.name).symlink_to(path)
    for path in (public / "views").iterdir():
        if path.name != "index.html":
            (views / path.name).symlink_to(path)
    index = (public / "views/index.html").read_text(encoding="utf-8")
    marker = "</body>"
    if marker not in index:
        raise ValueError("Grafana index template is missing </body>")
    (views / "index.html").write_text(
        index.replace(
            marker,
            '<script nonce="[[.Nonce]]" src="/public/mock-view.js"></script>\n'
            + marker,
            1,
        ),
        encoding="utf-8",
    )
    shutil.copyfile(
        Path(__file__).with_name("grafana-frame.js"), static_root / "mock-view.js"
    )
    return static_root


def prepare_dashboard(value):
    validate_dashboard(value)
    result = copy.deepcopy(value)
    result.pop("annotation_steps", None)
    result.update(uid="mock-experiment", id=None, editable=False, refresh="")

    def bind(panels):
        for panel in panels:
            if panel.get("type") == "row":
                bind(panel.get("panels", []))
                continue
            panel["datasource"] = dict(DATASOURCE)
            for target in panel.get("targets", []):
                target["datasource"] = dict(DATASOURCE)

    bind(result["panels"])
    for variable in result.get("templating", {}).get("list", []):
        if variable.get("type") == "query":
            variable["datasource"] = dict(DATASOURCE)
    result["annotations"] = {
        "list": [
            {
                "datasource": {"type": "grafana", "uid": "-- Grafana --"},
                "enable": True,
                "hide": True,
                "iconColor": "#ff9830",
                "name": "Step / 门禁",
                "type": "tags",
                "tags": ["$annotation_steps"],
                "matchAny": True,
                "limit": 10000,
            }
        ]
    }
    return result


def provision(
    directory, value, prometheus_url, start, end, *, scrape_interval, filters, steps=()
):
    selected = value.get("annotation_steps", list(steps))
    if not isinstance(selected, list) or any(
        not isinstance(step, str) for step in selected
    ):
        raise ValueError("view.annotation_steps must be a list of Step IDs")
    if len(set(selected)) != len(selected) or set(selected) - set(steps):
        raise ValueError("view.annotation_steps contains duplicate or unknown Step IDs")
    for path in ("dashboards", "provisioning/dashboards", "provisioning/datasources"):
        (directory / path).mkdir(parents=True)
    value = prepare_dashboard(value)
    variables = value.setdefault("templating", {}).setdefault("list", [])
    if any(variable.get("name") == "annotation_steps" for variable in variables):
        raise ValueError("annotation_steps is a reserved View variable")
    options = [{"text": "不显示", "value": "__none__", "selected": not selected}]
    options += [
        {"text": step, "value": "step:" + step, "selected": step in selected}
        for step in steps
    ]
    variables.append(
        {
            "name": "annotation_steps",
            "label": "事件线 Step",
            "type": "custom",
            "multi": True,
            "includeAll": True,
            "query": ",".join(
                option["text"] + " : " + option["value"] for option in options
            ),
            "options": options,
            "current": {
                "text": selected or ["不显示"],
                "value": ["step:" + step for step in selected] or ["__none__"],
            },
        }
    )
    # Freeze options from the full experiment, including scaled-in instances.
    for variable in value.get("templating", {}).get("list", []):
        if variable["name"] in filters:
            addresses = filters[variable["name"]]
            variable.update(
                type="custom",
                query=",".join(addresses),
                options=[
                    {"text": address, "value": address, "selected": False}
                    for address in addresses
                ],
            )
            if not addresses:
                variable["hide"] = 2
            variable.pop("datasource", None)
            variable.pop("definition", None)
    value["time"] = {
        key: datetime.fromtimestamp(value / 1000, timezone.utc).isoformat()
        for key, value in [("from", start), ("to", end)]
    }
    (directory / "dashboards/default.json").write_text(
        json.dumps(value, ensure_ascii=False), encoding="utf-8"
    )
    embedded = copy.deepcopy(value)
    embedded["uid"] = "mock-experiment-embedded"
    for variable in embedded.get("templating", {}).get("list", []):
        if variable["name"] in filters or variable["name"] == "annotation_steps":
            variable["hide"] = 2  # Hide only the controls owned by the parent page.
    (directory / "dashboards/embedded.json").write_text(
        json.dumps(embedded, ensure_ascii=False), encoding="utf-8"
    )

    def write(relative, content):
        (directory / relative).write_text(yaml.safe_dump(content), encoding="utf-8")

    write(
        "provisioning/datasources/default.yaml",
        {
            "apiVersion": 1,
            "datasources": [
                {
                    **DATASOURCE,
                    "name": "Experiment Prometheus",
                    "access": "proxy",
                    "url": prometheus_url,
                    "isDefault": True,
                    "jsonData": {"timeInterval": scrape_interval},
                }
            ],
        },
    )
    write(
        "provisioning/dashboards/default.yaml",
        {
            "apiVersion": 1,
            "providers": [
                {
                    "name": "experiment",
                    "options": {"path": str(directory / "dashboards")},
                }
            ],
        },
    )
    return {"steps": list(steps), "selected": selected}


def instance_filters(url, start, end, view):
    """Use saved series, not current discovery or a new instance-history ledger."""
    variables = {
        variable["name"] for variable in view.get("templating", {}).get("list", [])
    }
    options = {
        name: set()
        for name in (
            "prefill_instance",
            "decode_instance",
            "vit_instance",
            "master_instance",
            "client_instance",
        )
        if name in variables
    }
    if not options:
        return {}
    query = urlencode(
        {"match[]": '{job="mock"}', "start": start / 1000, "end": end / 1000}
    )
    with urlopen(url + "/api/v1/series?" + query, timeout=10) as response:
        result = json.load(response)
    if (
        result.get("status") != "success"
        or result.get("warnings")
        or not isinstance(result.get("data"), list)
    ):
        raise ValueError("Prometheus failed to enumerate experiment instances")
    for series in result["data"]:
        name, address = series.get("__name__", ""), series.get("instance")
        if not isinstance(name, str) or not isinstance(address, str) or not address:
            raise ValueError("Invalid Prometheus series identity")
        role = series.get("role")
        if role in ("prefill", "decode", "vit"):
            variable = role + "_instance"
        elif name.startswith("mock_client_"):
            variable = "client_instance"
        elif name.startswith("flexlb_") or series.get("application") == "flexlb-api":
            variable = "master_instance"
        else:
            continue
        if variable in options:
            options[variable].add(address)
    return {name: sorted(addresses) for name, addresses in options.items()}


def post_annotations(url, password, events, cancelled):
    authorization = "Basic " + base64.b64encode(("admin:" + password).encode()).decode()
    for event in events:
        if cancelled.is_set():
            return
        body = {
            "time": event["time"],
            "text": escape(event["text"]).replace("\n", "<br/>"),
            "tags": ["experiment", "step:" + event["step_id"]],
        }
        if event.get("timeEnd") is not None:
            body["timeEnd"] = event["timeEnd"]
        request = Request(
            url + "/api/annotations",
            json.dumps(body).encode(),
            {"Content-Type": "application/json", "Authorization": authorization},
        )
        with urlopen(request, timeout=3) as response:
            json.load(response)
