package org.flexlb.mock.metrics;

import io.prometheus.client.Collector.MetricFamilySamples;
import io.prometheus.client.CollectorRegistry;
import io.prometheus.client.exporter.common.TextFormat;
import java.io.*;
import java.util.*;

/** Physical Whale metadata; Prometheus owns job/instance, never the Engine. */
public final class MetricTags {
    private MetricTags() { }

    public static Map<String, String> fromEnvironment(Map<String, String> environment) {
        var tags = new LinkedHashMap<String, String>();
        if (!environment.getOrDefault("HIPPO_ROLE", "").isBlank()) {
            for (var pair : Map.of("hippo_app", "HIPPO_APP", "hippo_role", "HIPPO_ROLE",
                                   "hippo_group", "HIPPO_SERVICE_NAME", "host_ip", "HIPPO_SLAVE_IP").entrySet()) {
                var value = environment.get(pair.getValue());
                if (value != null && !value.isBlank()) tags.put(pair.getKey(), value);
            }
            var container = environment.getOrDefault("RequestedIP", environment.getOrDefault("HIPPO_SLAVE_IP", ""));
            if (!container.isBlank()) tags.put("container_ip", container);
        }
        return Map.copyOf(tags);
    }

    public static String scrape(CollectorRegistry registry, Map<String, String> tags) {
        var ordered = new TreeMap<>(tags);
        var families = new ArrayList<MetricFamilySamples>();
        var source = registry.metricFamilySamples();
        while (source.hasMoreElements()) {
            var family = source.nextElement();
            var samples = new ArrayList<MetricFamilySamples.Sample>();
            for (var sample : family.samples) {
                var names = new ArrayList<>(sample.labelNames);
                var values = new ArrayList<>(sample.labelValues);
                for (var tag : ordered.entrySet()) {
                    if (names.contains(tag.getKey())) throw new IllegalArgumentException("Duplicate metric tag: " + tag.getKey());
                    names.add(tag.getKey()); values.add(tag.getValue());
                }
                samples.add(new MetricFamilySamples.Sample(sample.name, names, values, sample.value, sample.exemplar, sample.timestampMs));
            }
            families.add(new MetricFamilySamples(family.name, family.unit, family.type, family.help, samples));
        }
        var text = new StringWriter();
        try { TextFormat.write004(text, Collections.enumeration(families)); }
        catch (IOException impossible) { throw new UncheckedIOException(impossible); }
        return text.toString();
    }
}
