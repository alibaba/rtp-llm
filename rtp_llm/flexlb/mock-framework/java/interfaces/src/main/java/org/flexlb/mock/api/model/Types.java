package org.flexlb.mock.api.model;

import java.util.List;
import java.util.Map;

/** Java instance protocol values; Case contracts live in Python. */
public final class Types {
    private Types() {}
    public enum Role { prefill, decode, vit }
    public record None() {}
    public record MasterRef(String id) {}
    public record MasterSet(List<MasterRef> instances) {}
    public record PerMasterEngines(Map<String, List<String>> instances) {}
    public record MasterReceipt(MasterSet affected) {}
    public record MetricValue(double value, String unit) {}
    public record QuerySample(Map<String, String> labels, double value, String sample_time) {}
    public record QueryResult(List<QuerySample> samples, String query_time) {}
    public record EngineSelection(Role role) {}
}
