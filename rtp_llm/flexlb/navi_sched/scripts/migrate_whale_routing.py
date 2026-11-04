#!/usr/bin/env python3
"""Move NaviSched discovery settings from a shared template into deployments."""

import argparse
import copy
import http.cookiejar
import json
import os
import urllib.parse
import urllib.request


ROUTING_ENVS = {
    "VLLM_NAVI_SCHED_MASTER_ADDRESSES",
    "VLLM_NAVI_SCHED_MASTER_BIZ_NAME",
    "VLLM_NAVI_SCHED_MASTER_VIP_DOMAIN",
    "VLLM_NAVI_SCHED_BIZ_NAME",
    "VLLM_NAVI_SCHED_TARGET_BIZ_NAME",
    "NAVI_SCHED_MASTER_CONFIG_JSON",
}
DEFAULT_MASTER_BIZ_NAME = "vllm_navi_sched_master"
API = "https://whale.online.alibaba-inc.com/proxy/whale/api/v3"


def env_map(items):
    return {item["name"]: item.get("value", "") for item in items or []} if items and isinstance(items[0], dict) else dict(items or [])


def clean_template(content):
    result = copy.deepcopy(content)

    def visit(value):
        if isinstance(value, dict):
            for name, child in value.items():
                if name in ("envs", "default_process_envs"):
                    value[name] = [item for item in child if item[0] not in ROUTING_ENVS]
                    for item in value[name]:
                        if item[0] == "VLLM_NAVI_SCHED_ENABLE":
                            item[1] = "0"
                else:
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(result)
    return result


def merge_process(zone, additions):
    setting = copy.deepcopy(zone.get("zone_process_setting") or {})
    process = setting.setdefault("process_info", {})
    values = env_map(process.get("envs"))
    values.update(additions)
    process["envs"] = sorted(values.items())
    return {"zone_name": zone["zone_name"], "zone_process_setting": setting}


def plan_deployment(deployment, template, master_domain, agent_domain, private_discovery,
                    master_biz_name=DEFAULT_MASTER_BIZ_NAME):
    content = template["content"]
    if isinstance(content, str):
        content = json.loads(content)
    prefix = "com.aicheng.whale.prod." + deployment["biz_name"].lower() + "."
    if not master_domain.startswith(prefix) or not agent_domain.startswith(prefix):
        raise ValueError("discovery domains must belong to the deployment's service")
    if not master_biz_name:
        raise ValueError("Master biz name must not be empty")
    processes = []
    discoveries = []
    for zone in deployment["zone_deployments"]:
        if zone["zone_type"] == "PY_MASTER":
            registered = {item.get("domain") for item in zone.get("service_discoveries", [])}
            if master_domain not in registered:
                raise ValueError("Master domain is not registered by this deployment")
            base = content["master_zone_plan"]
            values = env_map(base.get("default_process_envs"))
            resources = base["zone_resource_templates"]
            if len(resources) != 1:
                raise ValueError("expected one Master resource template")
            values.update(env_map(resources[0]["process_info"].get("envs")))
            values.update(env_map(zone.get("zone_process_setting", {}).get("process_info", {}).get("envs")))
            config = json.loads(values.get("NAVI_SCHED_MASTER_CONFIG_JSON", "{}"))
            runtime = config.setdefault("runtime_config", {})
            runtime.pop("local_nodes", None)
            subscription = runtime.setdefault("vip_subscription", {})
            subscription.update(services=[agent_domain], grpc_port_diff=4)
            subscription.setdefault("jmenv_domain", "jmenv.tbsite.net")
            config.setdefault("kmon_tags", {}).update(deployment_name=deployment["deployment_name"], role="master")
            processes.insert(0, merge_process(zone, {
                "NAVI_SCHED_MASTER_CONFIG_JSON": json.dumps(config, separators=(",", ":"), sort_keys=True),
            }))
        elif zone["zone_type"] == "PY_INFERENCE":
            processes.append(merge_process(zone, {
                "VLLM_NAVI_SCHED_ENABLE": "1",
                "VLLM_NAVI_SCHED_MASTER_ADDRESSES": "",
                "VLLM_NAVI_SCHED_MASTER_BIZ_NAME": master_biz_name,
                "VLLM_NAVI_SCHED_MASTER_VIP_DOMAIN": master_domain,
                "VLLM_NAVI_SCHED_BIZ_NAME": agent_domain,
                "VLLM_NAVI_SCHED_TARGET_BIZ_NAME": agent_domain,
            }))
            if private_discovery:
                services = copy.deepcopy(zone.get("service_discoveries", []))
                matches = [item for item in services if item.get("domain") == agent_domain]
                if matches:
                    for item in matches:
                        item["enable"] = True
                else:
                    source = next(item for item in services if item["type"] == "vip")
                    service = copy.deepcopy(source)
                    service.update(domain=agent_domain, enable=True)
                    services.append(service)
                discoveries.append({"zone_name": zone["zone_name"], "service_discoveries": services})
    if not processes:
        raise ValueError("deployment has no NaviSched zones")
    cleaned = clean_template(content)
    return {"processes": processes, "discoveries": discoveries, "template_content": cleaned}


class Whale:
    def __init__(self, cookie_file, project, employee):
        jar = http.cookiejar.MozillaCookieJar(cookie_file)
        jar.load(ignore_discard=True, ignore_expires=True)
        self.opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(jar))
        self.headers = {"projectName": project, "x-user-empid": employee,
                        "X-Custom-Referrer": "__namespace_id=rtp;__user_emp_id=" + employee}

    def request(self, method, path, body=None):
        headers = dict(self.headers)
        data = None
        if body is not None:
            data = json.dumps(body, separators=(",", ":")).encode()
            headers["Content-Type"] = "application/json"
        url = path if path.startswith("https://") else API + path
        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        with self.opener.open(request, timeout=45) as response:
            result = json.load(response)
        if result.get("success") is False:
            raise RuntimeError(json.dumps(result, ensure_ascii=False))
        return result.get("data", result)

    def snapshot(self, deployment):
        query = urllib.parse.urlencode({
            "page_num": 0, "page_size": 100, "sort_by": "modified_time@DESC",
            "search_condition": json.dumps({"deployment_id": deployment["id"]}),
        })
        snapshots = self.request("GET", "https://whale-api.alibaba-inc.com/api/v3/internal/raw/"
                                 "PY_BIZ_DEPLOYMENT_VERSION_SNAPSHOT?" + query)
        matches = [item for item in snapshots
                   if str(item["biz_version"]) == str(deployment["biz_version"])]
        if not matches:
            raise RuntimeError("published deployment snapshot was not found")
        return matches[0]


def publish_clean_template(api, deployment, snapshot, source_template, content):
    biz_id = deployment["py_biz_id"]
    latest = api.request("GET", "/py_biz_distribute/" + biz_id + "/latest_version_content")
    candidate_id = latest["template_info"]["template_version_id"]
    candidate = api.request("GET", "/py_biz_template_version/" + candidate_id)
    if candidate["template_id"] == source_template["template_id"] and candidate["content"] == content:
        template_id, target = candidate_id, candidate
    else:
        created = api.request("POST", "/py_biz_template/" + source_template["template_id"] + "/create_version", {
            "description": "NaviSched discovery belongs to deployment settings; no shared routing defaults",
            "content": content, "weight_convert_version": source_template["weight_convert_version"],
        })
        template_id = created if isinstance(created, str) else created.get("id") or created.get("template_version_id")
        if not template_id:
            raise RuntimeError("template creation returned no version ID")
        target = api.request("GET", "/py_biz_template_version/" + template_id)
    if target["content"] != content:
        raise RuntimeError("created template differs from the validated content")
    print("TEMPLATE " + template_id, flush=True)
    if template_id == candidate_id and int(latest["py_biz_version"]) > int(snapshot["biz_version"]):
        version = int(latest["py_biz_version"])
    else:
        generated = api.request("POST", "/py_biz_distribute/" + biz_id + "/gen_version_base", {
            "model_path_resource": copy.deepcopy(snapshot["model_path_resource"]),
            "model_type": snapshot["model_type"], "base_envs": env_map(snapshot["biz_env_options"]),
            "template_info": {"template_version_id": template_id},
        })
        version = int(generated["biz_version"])
        latest = api.request("GET", "/py_biz_distribute/" + biz_id + "/latest_version_content")
    print("DRAFT " + str(version), flush=True)
    if int(latest["py_biz_version"]) != version or latest["template_info"]["template_version_id"] != template_id:
        raise RuntimeError("another draft superseded the generated version")
    # Drafts contain the model URI; published snapshots also include its resolved location.
    if latest["model_path_resource"]["resource_uri"] != snapshot["model_path_resource"]["resource_uri"]:
        raise RuntimeError("generated version changed model URI")
    for key in ("model_type", "lora_path_resources",
                "model_param_count_b", "pretrain_model_param_count"):
        if latest.get(key) != snapshot.get(key):
            raise RuntimeError("generated version changed " + key)
    envs = env_map(latest.get("base_envs"))
    envs.update(env_map(latest.get("sp_envs")))
    original_envs = env_map(snapshot["biz_env_options"])
    if any(original_envs.get(key) != value for key, value in envs.items()):
        raise RuntimeError("generated version changed model environment")
    choices = api.request("GET", "/py_biz_distribute/" + biz_id + "/deployments_publish")
    choice = next(item for item in choices if item["deployment_id"] == deployment["id"])
    payload = {"biz_version": version, "deployments_ratio": [{
        "deployment_id": deployment["id"], "publish_ratio": 100,
        "published_rows": int(choice["row_count"]),
    }]}
    validation = api.request("POST", "/py_biz_distribute/" + biz_id + "/publish_version_validate", payload)
    if validation.get("success") is not True:
        raise RuntimeError("publish validation failed: " + repr(validation))
    published = api.request("POST", "/py_biz_distribute/" + biz_id + "/publish_version", payload)
    if published.get("failed_deployments"):
        raise RuntimeError("publish failed: " + repr(published))
    return {"biz_version": version, "template_id": template_id, "template_version": target["version"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment-id", required=True)
    parser.add_argument("--master-domain", required=True)
    parser.add_argument("--agent-domain", required=True)
    parser.add_argument("--master-biz-name", default=DEFAULT_MASTER_BIZ_NAME)
    parser.add_argument("--private-discovery", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--cookie-file", default=os.path.expanduser("~/.whale-auth/cookies.txt"))
    parser.add_argument("--project", default="futurelab_d_project")
    parser.add_argument("--employee", default="103303")
    args = parser.parse_args()
    api = Whale(args.cookie_file, args.project, args.employee)
    path = "/py_biz_deployment/" + args.deployment_id
    deployment = api.request("GET", path)
    snapshot = api.snapshot(deployment)
    template = api.request("GET", "/py_biz_template_version/" + snapshot["template_info"]["template_version_id"])
    plan = plan_deployment(deployment, template, args.master_domain, args.agent_domain,
                           args.private_discovery, args.master_biz_name)
    print(json.dumps({"deployment": deployment["deployment_name"], "source_version": deployment["biz_version"],
                      "processes": plan["processes"], "discoveries": plan["discoveries"],
                      "apply": args.apply}, ensure_ascii=False), flush=True)
    if not args.apply:
        return
    current = api.request("GET", path)
    if current != deployment:
        raise RuntimeError("deployment changed while planning; rerun to inspect the latest state")
    for payload in plan["processes"]:
        zone = next(item for item in current["zone_deployments"] if item["zone_name"] == payload["zone_name"])
        if zone.get("zone_process_setting") == json.loads(json.dumps(payload["zone_process_setting"])):
            continue
        api.request("POST", path + "/update_process", payload)
        print("UPDATED_PROCESS " + payload["zone_name"], flush=True)
    for payload in plan["discoveries"]:
        zone = next(item for item in current["zone_deployments"] if item["zone_name"] == payload["zone_name"])
        if zone.get("service_discoveries") == payload["service_discoveries"]:
            continue
        api.request("POST", path + "/update_zone_service_discoveries", payload)
        print("UPDATED_DISCOVERY " + payload["zone_name"], flush=True)
    current = api.request("GET", path)
    zones = {item["zone_name"]: item for item in current["zone_deployments"]}
    for payload in plan["processes"]:
        actual = zones[payload["zone_name"]].get("zone_process_setting", {})
        if json.loads(json.dumps(payload["zone_process_setting"])) != actual:
            raise RuntimeError("deployment did not retain requested process settings")
    result = publish_clean_template(api, current, snapshot, template, plan["template_content"])
    print("PUBLISHED " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
