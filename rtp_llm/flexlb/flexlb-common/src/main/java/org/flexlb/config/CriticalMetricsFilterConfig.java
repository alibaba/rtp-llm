package org.flexlb.config;

import io.micrometer.core.instrument.Meter;
import io.micrometer.core.instrument.config.MeterFilter;
import io.micrometer.core.instrument.config.MeterFilterReply;
import org.flexlb.metric.MicrometerFlexMonitor;
import org.springframework.boot.autoconfigure.condition.ConditionalOnClass;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import javax.annotation.PostConstruct;
import java.util.Arrays;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * Metrics whitelist filter for the Micrometer reporting path.
 *
 * <p>Replaces the former {@code flexlb.monitor.mode=critical-only} toggle +
 * hardcoded {@code CRITICAL_METRICS} set with a single configurable
 * whitelist via {@code flexlbMonitorCriticalMetrics}:
 * <ul>
 *   <li><b>Empty or {@code "*"}</b> → report all metrics (no filtering)</li>
 *   <li><b>Comma-separated metric names</b> (without the {@code flexlb.}
 *       prefix) → only those metrics are registered/reported</li>
 * </ul>
 *
 * <p>Filtering is applied at two layers:
 * <ol>
 *   <li><b>Early-return allowlist</b> — via {@link MicrometerFlexMonitor#setAllowedMetrics(Set)},
 *       non-allowlisted metrics are skipped at register() and report() time before any
 *       Micrometer interaction, avoiding unnecessary object allocation and map lookups.</li>
 *   <li><b>MeterFilter</b> — a defensive {@code MeterRegistry} filter that denies any
 *       {@code flexlb.} prefixed meter not in the allowlist, in case a meter was registered
 *       before the allowlist took effect or via a different code path.</li>
 * </ol>
 *
 * <p>Non-{@code flexlb.} metrics (jvm.*, process.*, http.server.requests, etc.) are always allowed.
 *
 * <p>This config is active only when {@code flexlb.monitor.enabled=true} (or missing).
 * When monitoring is disabled ({@code flexlb.monitor.enabled=false}), {@link MonitorDisableConfig}
 * provides a deny-all filter instead.
 *
 * <p>The KMonitor path is unaffected — production always reports all metrics.
 */
@Configuration
@ConditionalOnClass(name = "io.micrometer.core.instrument.MeterRegistry")
@ConditionalOnProperty(name = "flexlb.monitor.enabled", havingValue = "true", matchIfMissing = true)
public class CriticalMetricsFilterConfig {

    private static final String METRIC_PREFIX = "flexlb.";

    private final ConfigService configService;
    private Set<String> allowedMetrics;

    public CriticalMetricsFilterConfig(ConfigService configService) {
        this.configService = configService;
    }

    @PostConstruct
    public void init() {
        String configValue = configService.loadBalanceConfig().getFlexlbMonitorCriticalMetrics();
        if (configValue == null || configValue.isBlank() || "*".equals(configValue.trim())) {
            // Empty or "*" → allow all (no filtering)
            allowedMetrics = null;
            MicrometerFlexMonitor.setAllowedMetrics(null);
        } else {
            // Comma-separated metric names (without flexlb. prefix) → whitelist
            allowedMetrics = Arrays.stream(configValue.split(","))
                    .map(String::trim)
                    .filter(s -> !s.isEmpty())
                    .collect(Collectors.toSet());
            MicrometerFlexMonitor.setAllowedMetrics(allowedMetrics);
        }
    }

    /**
     * Defensive MeterRegistry filter: allows all non-flexlb metrics, and only allows
     * flexlb metrics whose unprefixed name is in the configured allowlist.
     * When the allowlist is null (empty or "*" config), all metrics are allowed.
     */
    @Bean
    public MeterFilter criticalMetricsOnlyFilter() {
        final Set<String> allowed = allowedMetrics;
        if (allowed == null) {
            // No whitelist configured — allow all metrics (neutral, so other
            // filters in the chain can still express opinions).
            return new MeterFilter() {
                @Override
                public MeterFilterReply accept(Meter.Id id) {
                    return MeterFilterReply.NEUTRAL;
                }
            };
        }
        return new MeterFilter() {
            @Override
            public MeterFilterReply accept(Meter.Id id) {
                String name = id.getName();
                if (!name.startsWith(METRIC_PREFIX)) {
                    return MeterFilterReply.NEUTRAL;
                }
                String unprefixed = name.substring(METRIC_PREFIX.length());
                return allowed.contains(unprefixed)
                        ? MeterFilterReply.NEUTRAL : MeterFilterReply.DENY;
            }
        };
    }
}
