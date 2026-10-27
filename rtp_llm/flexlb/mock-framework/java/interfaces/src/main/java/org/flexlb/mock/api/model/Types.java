package org.flexlb.mock.api.model;

import java.util.List;
import java.util.Map;
import org.flexlb.mock.api.Argument;

/** Cross-component data only; no RTP or old framework types. */
public final class Types {
    private Types() {}
    public enum Role { prefill, decode, vit }
    public enum Stat { avg, max, p99 }
    public enum ArrivalMode { trace, uniform }
    public record None() {}
    public record Duration(String value) {}
    public sealed interface TimeRange permits All, Latest, Interval {}
    public record All() implements TimeRange {}
    public record Latest() implements TimeRange {}
    public record Interval(Duration start, Duration end) implements TimeRange {}
    public record MasterRef(String id) {}
    public record EngineRef(String id) {}
    public record MasterSet(List<MasterRef> instances) {}
    public record EngineSet(List<EngineRef> instances) {}
    public record PerMasterEngines(Map<String, EngineSet> instances) {}
    public record MasterReceipt(MasterSet affected) {}
    public record ScaleReceipt(EngineSet removed, EngineSet remaining) {}
    public record TrafficReceipt(String started_at) {}
    public record StopReceipt(String stopped_at) {}
    public record DrainReceipt(String drained_at) {}
    public record MetricValue(double value, String unit) {}
    public record QuerySample(Map<String, String> labels, double value, String sample_time) {}
    public record QueryResult(List<QuerySample> samples, String query_time) {}
    public record Start(MasterSet instances) {}
    public record Scale(Role role, @Argument(minimum=1) int count) {}
    public record EngineSelection(Role role) {}
    public record ClientRange(@Argument(ranges={"all", "latest", "interval"}) TimeRange range) {}
    public record EngineRange(@Argument(ranges={"latest", "interval"}) TimeRange range) {}
    public record CacheRange(
        @Argument(ranges={"latest", "interval"}) TimeRange range,
        @Argument(optional=true) EngineSet engines) {}
    public record Ttft(Stat stat, @Argument(ranges={"all", "latest", "interval"}) TimeRange range) {}
    public record Promql(String promql) {}
    public record Prepare(
        @Argument(optional=true) OutputTokens output_tokens,
        @Argument(optional=true) Integer priority) {}
    public record Input(String trace, @Argument(optional=true) Prepare prepare) {}
    public record Arrival(ArrivalMode mode,
        @Argument(optional=true, minimum=0.000001) Double qps,
        @Argument(optional=true, minimum=0.000001) Double speed,
        @Argument(optional=true) Duration ramp) {}
    public record Playback(
        @Argument(optional=true) Arrival arrival,
        @Argument(optional=true, minimum=1) Integer max_concurrency,
        @Argument(optional=true) Duration request_timeout,
        @Argument(optional=true, minimum=1) Integer count,
        @Argument(optional=true, minimum=1) Integer max_laps) {}
    public record Traffic(Input input, Playback playback) {}
}
