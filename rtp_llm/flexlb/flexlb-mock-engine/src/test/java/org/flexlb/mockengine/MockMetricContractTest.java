package org.flexlb.mockengine;

import java.nio.file.Path;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import static org.flexlb.mockengine.MockEngineTestSupport.*;
import static org.junit.jupiter.api.Assertions.*;

class MockMetricContractTest {
    @TempDir Path directory;

    @Test
    void unknownMetricsFailAndNewRoleSpecificMetricsAreFiltered() {
        var sink = (org.flexlb.metric.FlexMonitor) java.lang.reflect.Proxy.newProxyInstance(
                getClass().getClassLoader(), new Class<?>[]{org.flexlb.metric.FlexMonitor.class},
                (proxy, method, args) -> {
                    if (method.getName().equals("report")) fail("wrong-role metric was reported");
                    return null;
                });
        var monitor = new WhaleMockMonitor(sink);
        monitor.reportEvent(Map.of("rtp_llm_sp_estimate_tpot_us", 100), Map.of("role", "ROLE_TYPE_PREFILL"));
        monitor.reportEvent(Map.of("rtp_llm_prefill_worker_theory_cache_all_hit_ratio", 1),
                Map.of("role", "ROLE_TYPE_DECODE"));
        assertThrows(IllegalArgumentException.class, () -> monitor.reportEvent(
                Map.of("undeclared_metric", 1), Map.of("role", "ROLE_TYPE_DECODE")));
    }

    @Test
    void bothHttpModesExportDeclaredMetricsWithMatchingTypesAndRoles() throws Exception {
        try (var cluster = MockEngineTestCluster.start(performanceModel(directory, "10"), 63020, 1, 1)) {
            for (String path : new String[]{"/metrics", "/metrics?per_engine=true"}) {
                String body = httpGet(cluster.controlPort(), path);
                for (var metric : MockMetricContract.HTTP) {
                    assertTrue(body.contains("# TYPE " + metric.name() + " "
                            + metric.type().name().toLowerCase(java.util.Locale.ROOT)));
                    if (metric.type() == MockMetricContract.Type.HISTOGRAM) continue;
                    if (path.equals("/metrics") && metric.aggregation() == MockMetricContract.Aggregation.NONE) continue;
                    assertTrue(body.contains(metric.name() + "{"), metric.name());
                    for (String line : body.lines().filter(line -> line.startsWith(metric.name() + "{")).toList()) {
                        String role = line.contains("role=\"prefill\"") ? "prefill" : "decode";
                        assertTrue(metric.belongsTo(role), line);
                    }
                }
            }
            for (var service : new JavaMockEngineCluster.FastRpcService[]{cluster.prefill(0), cluster.decode(0)})
                service.whaleMetrics().keySet().forEach(MockMetricContract::require);
        }
    }
}
