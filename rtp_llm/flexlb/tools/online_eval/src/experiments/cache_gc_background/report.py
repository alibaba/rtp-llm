"""Summarize stable-topology GC calibration with the native report layer."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re
import statistics

from reporting.components import details, run_meta, table
from reporting.core import write_bundle


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stats(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    return dict(mean=statistics.mean(values), min=values[0], max=values[-1],
                p95=values[min(len(values)-1, int(len(values)*.95))])


def meter(row, fragment, labels=None):
    values = [m["value"] for m in row.get("meters", []) if fragment in m["name"]
              and all(m["labels"].get(k) == v for k, v in (labels or {}).items())]
    return sum(values) if values else None


def rates(rows, key):
    result = []
    for a, b in zip(rows, rows[1:]):
        if a.get(key) is not None and b.get(key) is not None and b[key] >= a[key]:
            result.append(dict(epoch_s=b["epoch_s"], value=(b[key]-a[key])/(b["epoch_s"]-a["epoch_s"])))
    return result


def rate_stats(rows, key):
    points = rates(rows, key)
    result = stats(p["value"] for p in points)
    valid = [r for r in rows if r.get(key) is not None]
    if result is not None and len(valid) > 1:
        if any(b[key] < a[key] for a,b in zip(valid,valid[1:])):
            return None
        result["mean"] = (valid[-1][key]-valid[0][key])/(valid[-1]["epoch_s"]-valid[0]["epoch_s"])
    return result


def gc_events(path):
    pattern = re.compile(r"^\[([^]]+)\].*?\[gc\s*\]\s+GC\((\d+)\) (Pause .+?)\s+(\d+[KMG])->(\d+[KMG])\([^)]*\) ([\d.]+)ms$")
    result = []
    if not path.exists():
        return result
    for line in path.read_text().splitlines():
        match = pattern.match(line)
        if match:
            stamp, identity, name, before, after, duration = match.groups()
            duration = float(duration)
            end = datetime.datetime.strptime(stamp, "%Y-%m-%dT%H:%M:%S.%f%z").timestamp()
            result.append(dict(epoch_s=end-duration/1000, pause_ms=duration,
                               full="Pause Full" in name, id=int(identity), name=name,
                               before=before, after=after))
    return result


def resource_series(path):
    values = [json.loads(line) for line in path.open()] if path.exists() else []
    for a,b in zip(values, values[1:]):
        elapsed = b["epoch_s"]-a["epoch_s"]
        b["cpu_cores"] = (b["cpu"]["usage_usec"]-a["cpu"]["usage_usec"])/elapsed/1e6
        b["throttled_s_per_s"] = (b["cpu"]["throttled_usec"]-a["cpu"]["throttled_usec"])/elapsed/1e6
    return values


def cpu_stats(rows):
    result = stats(r.get("cpu_cores") for r in rows)
    if result is not None and len(rows) > 1:
        result["mean"] = ((rows[-1]["cpu"]["usage_usec"] - rows[0]["cpu"]["usage_usec"])
                          / (rows[-1]["epoch_s"] - rows[0]["epoch_s"]) / 1e6)
    return result


def heap_bytes(text):
    return int(text[:-1]) * {"K":2**10,"M":2**20,"G":2**30}[text[-1]]


def summarize(root):
    manifest = json.loads((root/"manifest.json").read_text())
    events = json.loads((root/"events.json").read_text())
    zero = next(e["epoch_s"] for e in events if e["name"] == "cache_only_started")
    baseline = None
    with (root/"observations.jsonl").open() as source:
        for line in source:
            row = json.loads(line)
            if row["epoch_s"] > zero:
                break
            if (row.get("p_workers") == manifest["config"]["topology"]["prefill"]
                    and row.get("d_workers") == manifest["config"]["topology"]["decode"]):
                baseline = row
    if baseline is None:
        raise ValueError("missing stable pre-window worker baseline")
    identities = baseline["healthy_generations"]
    compact = []
    for row in (json.loads(line) for line in (root/"observations.jsonl").open()):
        output = {k:v for k,v in row.items() if k not in {"meters", "healthy_generations"}}
        current = row.get("healthy_generations", {})
        output["client_started"] = (row.get("client") or {}).get("started")
        available = "healthy_generations" in row and "error" not in row
        output["missing_workers"] = len(identities.keys()-current.keys()) if available else None
        output["replaced_workers"] = sum(identities[k] != current[k] for k in identities.keys() & current.keys()) if available else None
        for key, fragment, labels in [
            ("diff_scanned_keys", "cache_diff_scanned_keys_total", {"role":"PREFILL"}),
            ("diff_calls", "cache_diff_calls_total", {"role":"PREFILL"}),
            ("p_cache_sync_success", "cache_status_check_visitor_success_qps_total", {"role":"PREFILL"}),
            ("diff_added_sample", "cache_diff_added_blocks_size", {"role":"PREFILL"}),
            ("diff_removed_sample", "cache_diff_removed_blocks_size", {"role":"PREFILL"}),
            ("status_queue", "engine_balancing_thread_pool_info", {"threadPool":"statusCheckExecutor", "type":"queueSize"}),
            ("grpc_client_queue", "engine_balancing_thread_pool_info", {"threadPool":"gRpcExecutor", "type":"queueSize"}),
            ("engine_sync_queue", "engine_balancing_thread_pool_info", {"threadPool":"engineSyncExecutor", "type":"queueSize"}),
            ("client_eventloop_pending", "engine_balancing_event_loop_group_info", {"name":"gRpcEventLoopGroup", "type":"pending-task-total-count"}),
            ("worker_status_commits", "engine_health_check_visitor_success_qps_total", None),
            ("worker_rpc_calls", "grpc_call_count_total", {"service":"GetWorkerStatus", "retry":"false"}),
            ("worker_response_size_sample", "grpc_response_size", {"service":"GetWorkerStatus", "retry":"false"}),
        ]:
            output[key] = meter(row, fragment, labels)
        compact.append(output)
    intensity_path = root/"intensity.json"
    intensity = json.loads(intensity_path.read_text()) if intensity_path.exists() else {}
    gc_available = (root/"gc.log").is_file() and (root/"gc.log").stat().st_size > 0
    gc = gc_events(root/"gc.log")
    resources = resource_series(root/"resources.jsonl")
    mock_resources = resource_series(root/"mock-resources.jsonl")
    mock_gc_available = (root/"env/mock_engine_gc.log").is_file()
    mock_gc = gc_events(root/"env/mock_engine_gc.log")
    summary = {}
    for phase in ("cache_only", "traffic_warmup", "measurement"):
        group = [r for r in compact if r["phase"] == phase]
        if len(group) < 2:
            continue
        start, end = group[0]["epoch_s"], group[-1]["epoch_s"]
        pauses = [g for g in gc if start <= g["epoch_s"]+g["pause_ms"]/1000 <= end]
        overlap = lambda g: max(0, min(end, g["epoch_s"]+g["pause_ms"]/1000)-max(start,g["epoch_s"]))
        pause_seconds = sum(overlap(g) for g in gc)
        full_pause_seconds = sum(overlap(g) for g in gc if g["full"])
        res = [r for r in resources if start <= r["epoch_s"] <= end]
        mock_res = [r for r in mock_resources if start <= r["epoch_s"] <= end]
        mock_pauses = [g for g in mock_gc if start <= g["epoch_s"] <= end]
        summary[phase] = dict(sample_window_s=end-start, samples=len(group),
            max_gap_s=max(b["epoch_s"]-a["epoch_s"] for a,b in zip(group,group[1:])),
            gc_count=len(pauses) if gc_available else None, full_gc_count=sum(g["full"] for g in pauses) if gc_available else None,
            gc_pause_s=pause_seconds if gc_available else None,
            full_gc_per_min=sum(g["full"] for g in pauses)*60/(end-start) if gc_available else None,
            full_gc_pause_s=full_pause_seconds if gc_available else None,
            full_gc_max_pause_ms=max((g["pause_ms"] for g in pauses if g["full"]), default=0) if gc_available else None,
            gc_pause_fraction=pause_seconds/(end-start) if gc_available else None,
            cache_rpc=intensity.get(phase),
            **{key:stats(r.get(key) for r in group) for key in [
                "scheduler_inflight", "p_running_mean", "d_running_mean", "global_keys", "max_status_age_ms",
                "missing_workers", "replaced_workers", "stale_workers", "reported_dead", "heap_bytes", "direct_bytes",
                "status_queue", "grpc_client_queue", "engine_sync_queue", "client_eventloop_pending", "worker_response_size_sample"]},
            mock_cpu_cores=cpu_stats(mock_res),
            mock_gc_pause_s=sum(g["pause_ms"] for g in mock_pauses)/1000 if mock_gc_available else None,
            mock_gc_pause_fraction=sum(g["pause_ms"] for g in mock_pauses)/1000/(end-start) if mock_gc_available else None,
            mock_full_gc_count=sum(g["full"] for g in mock_pauses) if mock_gc_available else None,
            cpu_cores=cpu_stats(res),
            memory_bytes=stats(r.get("memory_bytes") for r in res),
            rss_bytes=stats(r.get("rss_bytes") for r in res),
            rates={key:rate_stats(group,key) for key in [
                "process_allocated_bytes", "diff_scanned_keys", "diff_calls", "worker_rpc_calls", "p_cache_sync_success", "worker_status_commits", "client_started"]})
    whole_end = next((e["epoch_s"] for e in events if e["name"] == "measurement_finished"), compact[-1]["epoch_s"])
    whole_gc = [g for g in gc if zero <= g["epoch_s"] + g["pause_ms"]/1000 <= whole_end]
    observation_summary = dict(window="cache_only_started through measurement_finished", duration_s=whole_end-zero,
        full_gc_count=sum(g["full"] for g in whole_gc) if gc_available else None,
        full_gc_per_min=sum(g["full"] for g in whole_gc)*60/(whole_end-zero) if gc_available else None,
        full_gc_max_pause_ms=max((g["pause_ms"] for g in whole_gc if g["full"]), default=0) if gc_available else None)
    checks = []
    measured = summary.get("measurement", {})
    for key, limits in manifest["config"]["targets"].items():
        if key in {"recent_keys_per_s", "p_full_responses_per_s"}:
            continue  # Source audit disproves the old metric interpretation.
        actual = ((measured.get("cache_rpc") or {}).get("responses") if key == "p_full_responses_per_s"
                  else ((measured.get("rates",{}).get(key.removesuffix("_per_s")) or {}).get("mean") if key.endswith("_per_s") else (measured.get(key) or {}).get("mean")))
        values = [r.get(key) for r in compact if r["phase"] == "measurement" and r.get(key) is not None]
        checks.append(dict(metric=key, expected=limits, actual=actual,
                           samples_within_fraction=(sum(limits[0] <= v <= limits[1] for v in values)/len(values)) if values else None,
                           matched=actual is not None and limits[0] <= actual <= limits[1]))
    errors = json.loads((root/"errors.json").read_text()) if (root/"errors.json").exists() else ["missing terminal errors receipt"]
    client_path = root/"client-evidence.json"
    client = json.loads(client_path.read_text()) if client_path.exists() else None
    jfr_path = root/"jfr-summary.json"
    profile = json.loads(jfr_path.read_text()) if jfr_path.exists() else None
    probes = []
    for name in ("healthy-worker-probes.jsonl", "healthy-worker-missing-probes.jsonl", "dead-event-live-probes.jsonl"):
        probes_path = root/name
        if probes_path.exists():
            probes.extend(json.loads(line) for line in probes_path.open())
    rejected_lines = None
    oom_lines = None
    if (root/"env/mock_engine.log").exists():
        rejected_lines = oom_lines = 0
        with (root/"env/mock_engine.log").open(errors="replace") as mock_log:
            for line in mock_log:
                rejected_lines += "RejectedExecutionException" in line
                oom_lines += "OutOfMemoryError" in line
    manifest["verified_request_plan_sha256"] = hashlib.sha256((root/"traffic.jsonl").read_bytes()).hexdigest()
    launched = {kind: json.loads((root / (kind + "-launched-artifact.json")).read_text())
                for kind in ("master", "mock") if (root / (kind + "-launched-artifact.json")).exists()}
    artifact_verified = (launched.get("master", {}).get("sha256") == manifest["master_sha256"]
                         and launched.get("mock", {}).get("sha256") == manifest["mock_sha256"]
                         and manifest["verified_request_plan_sha256"] == manifest["request_plan_sha256"])
    consistency = json.loads((root/"cache-consistency.json").read_text()) if (root/"cache-consistency.json").exists() else None
    terminal = {e["name"] for e in events}
    valid = (artifact_verified and "observation_finished" in terminal and not errors
             and bool(consistency and consistency.get("consistent"))
             and rejected_lines == 0 and oom_lines == 0)
    return dict(observation_summary=observation_summary, status="completed" if "observation_finished" in terminal else "incomplete",
                outcome=dict(gate="OBSERVATION_ONLY", validity="VALID" if valid else "INSUFFICIENT_OR_INVALID"),
                artifact_verified=artifact_verified, launched_artifacts=launched, cache_consistency=consistency, manifest=manifest, events=events, frames=compact, resources=resources, mock_resources=mock_resources, mock_gc=mock_gc, gc=gc, profile=profile, probes=probes,
                summary=summary, calibration_checks=checks, errors=errors, client=client, gc_available=gc_available,
                mock_rejection_log_lines=rejected_lines, mock_oom_log_lines=oom_lines,
                raw_root=str(root), raw_observations_sha256=file_sha256(root/"observations.jsonl"),
                conclusion="背景匹配需逐项验收；出现Full GC不等于证明线上同因。",
                metric_correction="app.cache.recent.key.total.count is a GAUGE of per-request input tokens, not keys/s.")


def display(value, *, scale=1, suffix=""):
    return "未采集" if value is None else format(value*scale, ".5g") + suffix


def report(data, output, identity):
    rows = data["frames"]
    start = next(e["epoch_s"] for e in data["events"] if e["name"] == "cache_only_started")
    rows = [r for r in rows if r["epoch_s"] >= start]
    labels = dict(small_diff_started="开始百级小幅diff", cache_only_started="纯缓存背景开始", traffic_warmup_started="开始真实请求",
                  measurement_started="预热结束，开始测量", observation_finished="观测结束", consistency_started="冻结后核对缓存", consistency_finished="缓存核对完成", performance_observation_finished="性能窗口结束", measurement_finished="测量结束",
                  sending_stopped="停止发送新请求", drain_finished="请求排空完成")
    events = [dict(id=e["name"],name=labels.get(e["name"],e["name"]),t=e["epoch_s"]-start) for e in data["events"] if e["epoch_s"] >= start]
    end = max([r["epoch_s"] for r in rows]+[e["epoch_s"] for e in data["events"]])
    first_full = next((g for g in data["gc"] if g["full"] and g["epoch_s"] >= start), None)
    if first_full:
        events.append(dict(id="first_full_gc",name="首次Full GC（观测）",t=first_full["epoch_s"]-start))
    first_affected = next((r for r in rows if any((r.get(k) or 0)>0 for k in ("missing_workers","replaced_workers","stale_workers","reported_dead"))), None)
    if first_affected:
        events.append(dict(id="first_worker_affected",name="首次Worker缺失/失鲜/代际变化（观测）",t=first_affected["epoch_s"]-start))
    events.sort(key=lambda e:e["t"])
    consistency = data.get("cache_consistency") or {}
    feeder_known = data.get("mock_rejection_log_lines") is not None and data.get("mock_oom_log_lines") is not None
    feeder_valid = feeder_known and data["mock_rejection_log_lines"] == 0 and data["mock_oom_log_lines"] == 0
    measured = data["summary"].get("measurement", {})
    allocation = ((measured.get("rates") or {}).get("process_allocated_bytes") or {}).get("mean")
    conclusion = ("测量窗：分配" + display(allocation, scale=1/2**30, suffix=" GiB/s")
                  + "，Full GC " + display(measured.get("full_gc_per_min"), suffix="次/分钟")
                  + "；缓存一致性" + ("通过。" if consistency.get("consistent") else "未通过或缺少终态证据。"))
    complete = data.get("client",{}).get("complete",False) if data.get("client") else False
    def series(name,key,unit,values=None,scale=1):
        values = rows if values is None else values
        return dict(name=name,metric_id=key,unit=unit,axis="y",points=[
            dict(x=r["epoch_s"]-start,y=r[key]*scale if r.get(key) is not None else None) for r in values],
            provenance=dict(source=data["raw_root"],measurement=key))
    def panel(key,title,unit,values,caption):
        return dict(id=key,title=title,type="line",timeX=True,axes={"y":dict(title=unit)},
                    series=values,events=events,caption=caption)
    phase_rate_rows = []
    for phase, stats_ in data["summary"].items():
        interval = [r for r in rows if r["phase"] == phase]
        if interval and stats_.get("cache_rpc"):
            for edge in (interval[0]["epoch_s"], interval[-1]["epoch_s"]):
                phase_rate_rows.append(dict(epoch_s=edge, responses=stats_["cache_rpc"]["responses"],
                    entries=stats_["cache_rpc"]["entries"], bytes=stats_["cache_rpc"]["key_payload_bytes"],
                    full_responses=stats_["cache_rpc"].get("full_responses"),
                    unchanged_responses=stats_["cache_rpc"].get("unchanged_responses"),
                    added_keys=stats_["cache_rpc"].get("added_keys")))
    panels = [
        panel("cache-responses","P 缓存响应实际速率","次/s",[series("全部响应","responses","次/s",phase_rate_rows),series("完整keys响应","full_responses","次/s",phase_rate_rows),series("同版本轻量响应","unchanged_responses","次/s",phase_rate_rows)],"按Mock逐响应计数差分，分阶段均值；不能用目标轮询频率代替。"),
        panel("cache-key-volume","P 响应实际条目量","key/s",[series("P响应条目","entries","key/s",phase_rate_rows)],"实际返回的key条目；同版本轻量响应为0条，不能用响应次数代替。"),
        panel("cache-key-bytes","P 全量响应key载荷实际字节量","MB/s",[series("key protobuf载荷","bytes","MB/s",phase_rate_rows,1/1e6)],"缓存fixture protobuf实际getSerializedSize累计，不含另外合并的少量容量字段、gRPC帧和网络协议开销。"),
        panel("age","Worker 最大状态年龄","ms",[series("最大状态年龄","max_status_age_ms","ms")],"目录缺失及代际替换见下图，不能用缺失后的低年龄掩盖问题。"),
        panel("workers","Worker 目录及代际","个",[series(name,key,"个") for name,key in [
            ("P目录数量","p_workers"),("D目录数量","d_workers"),("缺失","missing_workers"),("代际替换","replaced_workers"),("失鲜","stale_workers"),("reported_alive=false","reported_dead")]],"全程没有注入上下线事件。代际比较使用缓存基线身份。"),
        panel("tasks","真实任务与在途请求","个",[series(name,key,"个") for name,key in [
            ("scheduler inflight","scheduler_inflight"),("P运行任务均值","p_running_mean"),("D运行任务均值","d_running_mean")]],"运行任务按观测到的engineIp指标均值；与调度器在途对象数不同。"),
        panel("queues","回调及同步队列","个",[series(name,key,"个") for name,key in [
            ("statusCheckExecutor","status_queue"),("gRPC client executor","grpc_client_queue"),("engineSyncExecutor","engine_sync_queue"),("客户端EventLoop pending","client_eventloop_pending")]],"采样值，不表示拒绝瞬间的真实峰值。"),
        panel("heap","Master 堆及直接内存","GiB",[series("堆","heap_bytes","GiB",scale=1/2**30),series("直接内存","direct_bytes","GiB",scale=1/2**30)],"JVM实际使用量；堆上限64GiB、直接内存上限8GiB。"),
        panel("global-keys","全局唯一缓存 key","个",[series("全局key","global_keys","个")],"观测到的全局索引数量；初始化目标为2892万。"),
    ]
    for key,title,unit,scale in [("process_allocated_bytes","Master 实际分配速率","GiB/s",1/2**30),
                                 ("diff_scanned_keys","实际完成的 diff 扫描量","key/s",1),
                                 ("diff_calls","实际完成的 diff 次数","次/s",1),
                                 ("p_cache_sync_success","Master成功提交的P缓存同步","次/s",1),
                                 ("worker_rpc_calls","WorkerStatus RPC成功返回量","次/s",1),
                                 ("worker_status_commits","Master成功提交的Worker状态","次/s",1),
                                 ("client_started","客户端实际发送QPS","次/s",1)]:
        points = [dict(epoch_s=p["epoch_s"],value=p["value"]) for p in rates(rows,key)]
        rate_series = series(title,"value",unit,points,scale)
        rate_series["metric_id"] = key + "_per_s"
        panels.append(panel(key,title,unit,[rate_series],
            "累计计数相邻样本差分；缺失不按零处理。diff扫描为完成比较时old.size+new.size，包含全量重建。"))
    gc = [dict(epoch_s=g["epoch_s"],pause_ms=g["pause_ms"]) for g in data["gc"] if start <= g["epoch_s"] <= end]
    full = [dict(epoch_s=g["epoch_s"],pause_ms=g["pause_ms"]) for g in data["gc"] if g["full"] and start <= g["epoch_s"] <= end]
    all_gc_series = series("GC暂停","pause_ms","ms",gc)
    all_gc_series["metric_id"] = "gc_pause_ms"
    full_series = series("Full GC","pause_ms","ms",full)
    full_series["metric_id"] = "full_gc_pause_ms"
    panels.append(panel("gc","每次 GC 暂停","ms",[all_gc_series,full_series],"取gc.log中完成的Pause事件；不包含并发回收总历时。"))
    frequency_rows = [dict(epoch_s=r["epoch_s"], full_gc_count_60s=(
        sum(g["full"] and r["epoch_s"]-60 < g["epoch_s"]+g["pause_ms"]/1000 <= r["epoch_s"] for g in data["gc"])
        if r["epoch_s"]-start >= 60 else None)) for r in rows]
    panels.append(panel("full-gc-frequency", "Full GC 频率（近60秒）", "次/分钟",
        [series("近60秒Full GC次数", "full_gc_count_60s", "次/分钟", frequency_rows)],
        "滚动60秒内完成的Full GC事件数，开始不足60秒不输出；不是全部GC次数。"))
    panels.append(panel("clock-mutation", "实际key变化速率（固定时钟）", "新增key/s",
        [series("实际新增keys", "added_keys", "新增key/s", phase_rate_rows)],
        "每P每6000ms变化117个key，与RPC次数独立；返回跳过多个epoch时累计计入。"))
    after_gc = [dict(epoch_s=g["epoch_s"]+g["pause_ms"]/1000,heap_after=heap_bytes(g["after"])) for g in data["gc"] if start <= g["epoch_s"] <= end]
    panels.append(panel("gc-heap-after","每次暂停GC后的已用堆","GiB",[series("GC后已用堆","heap_after","GiB",after_gc,1/2**30)],"来自gc.log；不等同于仅存活对象大小。Full GC后的数值另见原始事件。"))
    mock_resources = [r for r in data.get("mock_resources",[]) if start <= r["epoch_s"] <= end]
    panels.append(panel("mock-cpu","Mock模拟器 CPU","核",[series("模拟器CPU","cpu_cores","核",mock_resources)],"独立模拟器资源；用于排除发送端成为瓶颈。"))
    mock_pauses = [dict(epoch_s=g["epoch_s"],mock_pause_ms=g["pause_ms"]) for g in data.get("mock_gc",[]) if start <= g["epoch_s"] <= end]
    panels.append(panel("mock-gc","Mock模拟器每次GC暂停","ms",[series("模拟器GC","mock_pause_ms","ms",mock_pauses)],"来自Mock JVM的gc.log，单独显示，不计入Master暂停。"))
    resources = [r for r in data["resources"] if start <= r["epoch_s"] <= end]
    panels.append(panel("cpu","Master 实际 CPU 用量","核",[series("cgroup CPU","cpu_cores","核",resources)],"cgroup cpu.stat累计usage_usec差分；配额32核。"))
    panels.append(panel("memory","Master 容器与进程内存","GiB",[series("容器memory.current","memory_bytes","GiB",resources,1/2**30),series("进程RSS","rss_bytes","GiB",resources,1/2**30)],"容器预算120000MiB；不是堆大小。"))
    calibration_names = dict(p_full_responses_per_s="P全量响应（Mock，次/s）",
        p_cache_sync_success_per_s="P成功同步（Master，次/s）",scheduler_inflight="调度器在途数",
        p_running_mean="P运行任务均值（每IP）",d_running_mean="D运行任务均值（每IP）",
        worker_rpc_calls_per_s="WorkerStatus RPC成功返回（次/s）")
    phase_names = dict(cache_only="纯缓存与小幅diff",traffic_warmup="加入业务，预热",measurement="业务测量窗")
    allocation_rows = []
    weights = (data.get("profile") or {}).get("allocation_sample_weights", {})
    for phase in ("cache_only", "measurement"):
        sampled = [(site, weight) for site, weight in weights.items() if site.startswith(phase + "|")]
        total = sum(weight for _, weight in sampled)
        for site, weight in sorted(sampled, key=lambda item: -item[1])[:6]:
            _, method, klass = site.split("|", 2)
            allocation_rows.append([phase_names[phase], method, klass, display(weight / total, scale=100, suffix="%")])
    phase_rows = []
    for phase, values in data["summary"].items():
        rate = values["rates"]
        phase_rows.append([phase_names.get(phase,phase),display(values["sample_window_s"],suffix="s"),
            display((values.get("cache_rpc") or {}).get("responses")),
            display((rate.get("p_cache_sync_success") or {}).get("mean")),values["full_gc_count"],
            display(values["gc_pause_fraction"],scale=100,suffix="%"),
            display((rate.get("process_allocated_bytes") or {}).get("mean"),scale=1/2**30),
            display((rate.get("diff_scanned_keys") or {}).get("mean"),scale=1e-6),
            display((values.get("scheduler_inflight") or {}).get("mean")),
            display((values.get("max_status_age_ms") or {}).get("max"),scale=.001)])
    config = data["manifest"]["config"]
    implementation = {k:v for k,v in data["manifest"].items() if k != "config"}
    implementation["engine_omit_unchanged_keys"] = config["cache"]["omit_unchanged_keys"]
    spec = dict(report_spec_schema_version=1,run_id=identity,title="同版本省略 keys · Full GC 优化对照",
        subtitle=(conclusion if feeder_valid else "模拟器供给有效性未通过核验，不能归因Master健康性。")+"纯缓存背景开始=0秒；无Worker上下线注入。",
        timeOriginEpochS=start,timeOriginLabel="缓存基线稳定后，纯缓存观察开始",timeAxis=dict(min=0,max=end-start+1),
        events=events,panels=panels,
        run_meta=run_meta(dict(id=identity,kind="run",case="cache_gc_background_diagnostic"),
            implementation=implementation,environment=dict(topology=config["topology"],resources=config["resources"]),
            workload={k:({key:value for key,value in config[k].items() if key != "omit_unchanged_keys"} if k == "cache" else config[k]) for k in ("cache","traffic","status_poll_interval_ms") if k in config},
            configuration={k:config[k] for k in ("diagnostic_monitoring","mock_runtime","windows","targets") if k in config},evidence=dict(raw_root=data["raw_root"],sha256=data["raw_observations_sha256"])),
        kpis=[dict(label="纯缓存至测量结束Full GC",value=(data.get("observation_summary") or {}).get("full_gc_count", "未采集")),
              dict(label="启动产物与输入哈希",value="一致" if data.get("artifact_verified") else "未通过"),
              dict(label="缓存一致性",value="通过" if consistency.get("consistent") else "未通过/未完成"),
              dict(label="客户端闭环",value="完整" if complete else "未完整"),
              dict(label="模拟器供给有效性",value="无拒绝/OOM日志" if feeder_valid else "存在拒绝/OOM干扰" if feeder_known else "未验证"),
              dict(label="健康Worker稳定性",value="观测到异常" if first_affected else "本窗未见异常"),
              dict(label="测量窗Full GC",value=data["summary"].get("measurement",{}).get("full_gc_count","缺失"))],
        sections=[table("直接现象：Full GC、CPU与内存",
                    ["阶段","FullGC次/分","FullGC最长ms","FullGC暂停s","全部GC暂停%","CPU核时间均值","容器内存峰值GiB","RSS峰值GiB"],
                    [[phase_names.get(p,p),display(s.get("full_gc_per_min")),display(s.get("full_gc_max_pause_ms")),
                      display(s.get("full_gc_pause_s")),display(s.get("gc_pause_fraction"),scale=100),
                      display((s.get("cpu_cores") or {}).get("mean")),display((s.get("memory_bytes") or {}).get("max"),scale=1/2**30),
                      display((s.get("rss_bytes") or {}).get("max"),scale=1/2**30)] for p,s in data["summary"].items()]),
                  details("终态缓存一致性核对（性能窗口外）", consistency),
                  table("背景参照（不是优化有效性门槛）",["指标","预设范围","实际均值","采样在范围比例","判断"],
                    [[calibration_names.get(c["metric"],c["metric"]),str(c["expected"]),display(c["actual"]),
                      display(c["samples_within_fraction"],scale=100,suffix="%"),
                      "未判定" if c["actual"] is None else "范围内" if c["matched"] else "未达标"] for c in data["calibration_checks"]]),
                  table("分阶段实测",["阶段","时长","P生成/s","P提交/s","FullGC","暂停占比","分配GiB/s","diff百万key/s","在途均值","最大年龄s"],phase_rows),
                  table("JFR 分配采样权重 Top 6（不是精确分配量或存活量）",["阶段","首个FlexLB栈帧","对象类型","采样权重占比"],allocation_rows),
                  details("本轮版本边界", dict(source=data["manifest"]["source"], launched_artifacts=data.get("launched_artifacts"),
                    note="版本、实际启动产物及响应语义以本轮 manifest 为准；比较前核对 Mock、请求计划、资源和配置。")),
                  details("实验方法与假设",dict(phases="稳定大缓存与小幅diff→加入真实请求→固定窗口测量；全程不注入上下线。",
                      duty="版本每6000ms按时钟推进，与轮询次数无关；change_every_responses是保留的旧配置，被固定时钟覆盖。两组记录实际变更量；优化后全量响应下降是预期收益。",
                      monitoring="本轮改前改后均开启相同FlexMonitor、JFR、健康快照、分配计数、cgroup和GC日志；本轮不是监控开关对照。",
                      validity="存在Mock执行器拒绝/OOM时，不得归因Master误判；日志匹配行数不是RPC拒绝计数。",
                      production_input="app.cache.recent.key.total.count为输入token的GAUGE，不能解释成key/s；当前请求分布并非事故流量复刻。")),
                  details("直连健康探测（记录日志与探测时间边界）",data.get("probes",[])),
                  details("采样与客户端闭环",dict(errors=data["errors"],client=data["client"],mock_rejection_log_lines=data.get("mock_rejection_log_lines"),mock_oom_log_lines=data.get("mock_oom_log_lines"))),
                  details("从纯缓存开始至测量结束（包含预热）",data.get("observation_summary")),
                  details("分阶段完整统计",data["summary"]),
                  details("指标口径更正",data["metric_correction"]),details("对齐边界",config["boundaries"])])
    priority = ["full-gc-frequency", "gc", "process_allocated_bytes", "cpu", "memory", "heap",
                "age", "workers", "cache-responses", "cache-key-volume", "cache-key-bytes",
                "diff_scanned_keys", "clock-mutation", "global-keys", "worker_rpc_calls",
                "worker_status_commits", "queues", "tasks"]
    unavailable = [p["title"] for p in panels
                   if not any(point.get("y") is not None for series_ in p["series"] for point in series_["points"])]
    spec["panels"] = [p for p in panels if p["title"] not in unavailable]
    spec["panels"].sort(key=lambda p: priority.index(p["id"]) if p["id"] in priority else len(priority))
    spec["summary_sections"] = [spec["sections"].pop(0)]
    if unavailable:
        spec["sections"].append(details("未采集的指标（不按零处理）", unavailable))
    return write_bundle(output,"run",identity,data,spec,producer="cache_gc_background",role="diagnostic")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--evidence-only",action="store_true")
    args = parser.parse_args()
    data = summarize(args.input)
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/"evidence.json").write_text(json.dumps(data,ensure_ascii=False))
    if not args.evidence_only:
        print(report(data,args.output,args.input.name))


if __name__ == "__main__":
    main()
