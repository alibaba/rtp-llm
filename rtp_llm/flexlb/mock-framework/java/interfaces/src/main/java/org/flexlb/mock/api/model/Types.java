package org.flexlb.mock.api.model;

import java.util.List;
import java.util.Map;
import org.flexlb.mock.api.Argument;

/** Java instance protocol values; Case contracts live in Python. */
public final class Types {
    private Types() {}
    public enum Role { prefill, decode, vit }
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
    public record MetricValue(double value, String unit) {}
    public record QuerySample(Map<String, String> labels, double value, String sample_time) {}
    public record QueryResult(List<QuerySample> samples, String query_time) {}
    public record EngineSelection(Role role) {}
    public record EngineRange(@Argument(ranges={"latest", "interval"}) TimeRange range) {}
    public record Prepare(
        @Argument(optional=true) OutputTokens output_tokens,
        @Argument(optional=true) Integer priority) {}
}
