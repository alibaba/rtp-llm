package org.flexlb.config;

import io.micrometer.core.instrument.Meter;
import io.micrometer.core.instrument.config.MeterFilter;
import io.micrometer.core.instrument.config.MeterFilterReply;
import lombok.extern.slf4j.Slf4j;
import org.flexlb.metric.MicrometerFlexMonitor;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import javax.annotation.PostConstruct;
import java.util.AbstractSet;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * Opt-in metric whitelist mode: when {@code flexlb.monitor.mode=whitelist} is
 * active, only {@code flexlb.} prefixed metrics matching the comma-separated
 * {@code flexlb.monitor.metric-whitelist} property (env
 * {@code FLEXLB_MONITOR_METRIC_WHITELIST} via relaxed binding) are
 * registered/reported. Unlike {@link CriticalMetricsFilterConfig}'s hardcoded
 * curated set, the whitelist is property-driven, so deployments (e.g. the
 * online_eval harness) can trim the exposition surface down to exactly the
 * series their collector consumes.
 *
 * <p>Matching semantics — entries are <b>prometheus-form prefixes or full
 * names</b> (dots as underscores, leading {@code flexlb_} kept), so the list
 * can be copied verbatim from the collector-side whitelist
 * ({@code MASTER_PROMETHEUS_PREFIXES} in {@code eval_collectors.py}). A meter
 * name matches an entry when its prometheus form starts with the entry; the
 * counter exposition form (micrometer appends {@code _total} on the
 * prometheus endpoint) is also tried, so a {@code ..._total} entry matches
 * the counter's micrometer name (which does not carry the suffix).
 *
 * <p>This config filters at the same two layers as critical-only:
 * <ol>
 *   <li><b>Early-return allowlist</b> — via {@link MicrometerFlexMonitor#setAllowedMetrics(Set)}
 *       with a prefix-aware {@link Set} view, non-matching metrics are skipped
 *       at register()/report() time before any Micrometer interaction.</li>
 *   <li><b>MeterFilter</b> — a defensive {@code MeterRegistry} filter denying
 *       any non-matching {@code flexlb.} meter (e.g. one registered before the
 *       allowlist took effect).</li>
 * </ol>
 *
 * <p>Fail-safe: a missing/blank whitelist <b>denies every {@code flexlb.}
 * metric</b> (with a WARN log) rather than exposing everything — a mis-
 * configured whitelist should fail closed. Non-{@code flexlb.} metrics
 * (jvm.*, process.*, ...) are always allowed, same as critical-only.
 */
@Slf4j
@Configuration
@ConditionalOnClass(name = "io.micrometer.core.instrument.MeterRegistry")
@ConditionalOnProperty(name = "flexlb.monitor.mode", havingValue = "whitelist")
public class WhitelistMetricsFilterConfig {

    private static final String METRIC_PREFIX = "flexlb.";
    private static final String PROM_PREFIX = "flexlb_";
    private static final String COUNTER_SUFFIX = "_total";

    private final List<String> whitelist;

    public WhitelistMetricsFilterConfig(
            @Value("${flexlb.monitor.metric-whitelist:}") String metricWhitelist) {
        this.whitelist = parseWhitelist(metricWhitelist);
        if (this.whitelist.isEmpty()) {
            log.warn("flexlb.monitor.mode=whitelist but flexlb.monitor.metric-whitelist is "
                    + "empty/blank: denying every flexlb.* metric (fail-safe; configure a "
                    + "comma-separated prefix list, e.g. FLEXLB_MONITOR_METRIC_WHITELIST)");
        }
    }

    /**
     * Splits the comma-separated property into trimmed non-empty entries.
     */
    static List<String> parseWhitelist(String property) {
        if (property == null) {
            return List.of();
        }
        return Arrays.stream(property.split(","))
                .map(String::trim)
                .filter(entry -> !entry.isEmpty())
                .collect(Collectors.toUnmodifiableList());
    }

    /**
     * Prefix-or-full-name match on the prometheus form of an <b>unprefixed</b>
     * micrometer meter name (e.g. {@code app.cache.hit.ratio} →
     * {@code flexlb_app_cache_hit_ratio}). The counter exposition form
     * ({@code ..._total}) is tried as well so whitelist entries written for
     * the prometheus endpoint match the underlying counter meter.
     */
    static boolean matches(List<String> whitelist, String unprefixedMeterName) {
        String promName = PROM_PREFIX + unprefixedMeterName.replace('.', '_');
        for (String entry : whitelist) {
            if (promName.startsWith(entry) || (promName + COUNTER_SUFFIX).startsWith(entry)) {
                return true;
            }
        }
        return false;
    }

    /**
     * Defensive MeterRegistry filter: allows all non-flexlb metrics, and only
     * allows flexlb metrics whose prometheus-form name matches the whitelist.
     */
    @Bean
    public MeterFilter whitelistMetricsFilter() {
        return new MeterFilter() {
            @Override
            public MeterFilterReply accept(Meter.Id id) {
                String name = id.getName();
                if (!name.startsWith(METRIC_PREFIX)) {
                    return MeterFilterReply.NEUTRAL;
                }
                return matches(whitelist, name.substring(METRIC_PREFIX.length()))
                        ? MeterFilterReply.NEUTRAL : MeterFilterReply.DENY;
            }
        };
    }

    /**
     * Sets the early-return allowlist on {@link MicrometerFlexMonitor}. The
     * set is a prefix-aware view: {@code contains} applies
     * {@link #matches(List, String)} instead of exact equality, keeping the
     * early-return layer semantics aligned with the {@link MeterFilter}.
     */
    @PostConstruct
    public void init() {
        MicrometerFlexMonitor.setAllowedMetrics(new PrefixAllowSet(whitelist));
    }

    /**
     * {@link Set} view over the whitelist entries whose {@code contains} is
     * the prefix-or-full-name match on the (unprefixed) micrometer name —
     * {@code MicrometerFlexMonitor}'s exact-match allowlist check becomes
     * prefix-aware without touching that class. Iteration/size expose the
     * raw entries (log messages print the entry count).
     */
    static final class PrefixAllowSet extends AbstractSet<String> {

        private final List<String> entries;

        PrefixAllowSet(List<String> entries) {
            this.entries = List.copyOf(entries);
        }

        @Override
        public boolean contains(Object o) {
            return o instanceof String && matches(entries, (String) o);
        }

        @Override
        public Iterator<String> iterator() {
            return entries.iterator();
        }

        @Override
        public int size() {
            return entries.size();
        }
    }
}
