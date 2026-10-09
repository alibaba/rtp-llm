"""Selectable request and archived Prometheus views; never affect gate decisions."""


from monitoring.session import archived_series
from reporting.catalog import PALETTE


def panel(directory, evidence, result, presentation=None):
    import json
    from reporting.view_config import view

    presentation = presentation or view("master_performance.yaml")

    lo = evidence.get("window", {}).get("start_epoch_ms", 0)
    duration = evidence.get("criteria", {}).get("measure_s", 1)
    curves, audit = [], []
    axes = {}
    for descriptor in presentation["panels"]:
        for axis, settings in descriptor["axes"].items():
            if axis in axes and axes[axis] != settings:
                raise ValueError("conflicting view axis: " + axis)
            axes[axis] = dict(settings)

    def add(metric_id, name, group, axis, points, description, hidden=True,
            unit=None, apply_style=True):
        style = presentation["curves"][metric_id]
        if apply_style:
            name, group, axis = (style[field] for field in ("name", "group", "axis"))
            scale = style.get("scale", 1)
            points = [(t, value * scale if value is not None else None)
                      for t, value in points]
            unit = style.get("unit", unit)
        curves.append(
            dict(
                curve_id=metric_id,
                metric_id=style["metric_id"],
                name=name,
                group=group,
                axis=axis,
                unit=axes[axis]["title"] if unit is None else unit,
                points=[dict(x=t, y=v) for t, v in points],
                hidden=hidden,
                color=style.get("color") or PALETTE[
                    list(presentation["curves"]).index(metric_id) % len(PALETTE)],
                description=description,
            )
        )

    from monitoring.metric_store import MetricStore
    store = MetricStore.read(directory)
    for identity, rows in store.document["metrics"].items():
        if not identity.startswith("request/") or identity not in presentation["curves"]:
            continue
        style = presentation["curves"][identity]
        for row in rows:
            add(identity, style["name"], style["group"], style["axis"],
                [(t-lo/1000, value) for t,value in row["points"]],
                "逐请求证据；cohort 和完成窗口由指标生产器冻结，不替代整窗门禁 p99", True)
            curves[-1]["provenance"] = row["provenance"]

    series, sources, gaps, errors = archived_series(directory, lo / 1000)
    for key, points in series.items():
        epoch, source, metric, label_json = key.split("/", 3)
        if metric == "up":
            continue
        labels = json.loads(label_json)
        from reporting.metric_binding import bindings
        identity = sources[key]["metric_id"]
        selected = bindings(presentation, identity, labels)
        residual = {k:v for k,v in labels.items() if k != "role"}
        for curve_id, style in selected:
            name, group, axis = (style[field] for field in ("name", "group", "axis"))
            if residual:
                name += " · " + ", ".join(f"{k}={v}" for k, v in sorted(residual.items()))
            if any(c["name"] == name for c in curves):
                name += " · epoch " + epoch
            scale = style.get("scale", 1)
            visible = [(t, v * scale if v is not None else None)
                       for t, v in points if 0 <= t <= duration]
            add(curve_id, name, group, axis, visible, sources[key]["promql"],
                not style.get("primary", False), style.get("unit"), False)
            curves[-1]["metric_id"] = identity
            curves[-1]["provenance"] = sources[key]
            audit.append(dict(name=name, samples=sum(v is not None for _,v in visible), **sources[key]))
    # Never open an empty chart when only request-level evidence survived.
    if not any(not c["hidden"] and any(p["y"] is not None for p in c["points"]) for c in curves):
        for c in curves:
            if c["group"] == "客户端吞吐": c["hidden"] = False
    presets = {"核心": [c["name"] for c in curves if not c["hidden"]]}
    for group in dict.fromkeys(c["group"] for c in curves):
        presets[group] = [c["name"] for c in curves if c["group"] == group]
    return dict(
        id="performance",
        title="性能与运行状态",
        timeX=True,
        axes=axes,
        series=curves,
        presets=presets,
        caption="Prefill TPS 按引擎/DP 汇总 priority，与线上 context TPS、with cache TPS 口径对应；不对引擎执行速率求集群总和。时间按测量起点对齐。Client 曲线来自逐请求证据，mock/master 曲线来自归档 Prometheus（具体查询见审计）。"
        + (" 本报告缺少监控归档，只有请求级曲线。" if not series else ""),
    ), dict(queries=audit, gaps=gaps, errors=errors, available=bool(series))


def report_panels(curves, criteria, presentation):
    """Show the four measured views; keep gate floors separate from measurements."""
    panels = []
    duration = criteria.get("measure_s", 1)
    floor_metrics = {
        "mock/rtp_llm_context_tps_engine_mean/P": "mock/rtp_llm_context_tps",
        "mock/rtp_llm_context_tps_with_cache_engine_mean/P": "mock/rtp_llm_context_tps_with_cache",
        "mock/rtp_llm_generate_tps_engine_mean/D": "mock/rtp_llm_generate_tps",
    }
    for descriptor in presentation["panels"]:
        selected = [dict(curve, hidden=False) for metric_id in descriptor["curve_ids"]
                    for curve in curves if curve["curve_id"] == metric_id]
        # A monitoring query may exist but contain only NaNs. Show an explicit
        # gap rather than a 0% line or an apparently valid empty panel.
        populated = [curve for curve in selected if any(
            point["y"] is not None for point in curve["points"])]
        missing = [presentation["curves"][metric_id]["name"]
                   for metric_id in descriptor["curve_ids"]
                   if not any(curve["curve_id"] == metric_id for curve in populated)]
        caption = descriptor["caption"] if populated else descriptor["empty_caption"]
        if populated and missing:
            caption += " 缺少有效曲线：" + "、".join(missing) + "。"
        if descriptor["id"] == "engine-tps":
            floors = criteria.get("engine_tps", {})
            for metric_id in descriptor["curve_ids"]:
                metric = floor_metrics[metric_id]
                if metric in floors:
                    source = next((curve for curve in selected if curve["curve_id"] == metric_id), None)
                    name = presentation["curves"][metric_id]["name"]
                    selected.append(dict(
                        curve_id=metric_id + "/gate_floor", source_type="configuration",
                        name=name + " 门禁线", group="门禁", axis="forward",
                        unit="执行 tok/s", color=source["color"] if source else presentation["curves"][metric_id].get("color") or PALETTE[len(selected) % len(PALETTE)],
                        dash=[6, 4], hidden=False,
                        points=[dict(x=t, y=floors[metric]) for t in (0, duration)],
                        description="场景配置中的绝对下界；不是实测值",
                    ))
        panels.append(dict(
            id=descriptor["id"], title=descriptor["title"], caption=caption,
            timeX=True, axes=descriptor["axes"], series=selected,
        ))
    return panels
