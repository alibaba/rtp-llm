package org.flexlb.mockengine;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.*;

class WhaleModeConfigurationTest {
    @TempDir Path directory;

    @Test
    void wrapperAcceptsLiteralJsonWithoutShellCommandQuoting() throws Exception {
        Path javaExecutable = directory.resolve("java");
        Files.writeString(javaExecutable, "#!/bin/sh\nprintf '%s\\n' \"$@\"\n");
        assertTrue(javaExecutable.toFile().setExecutable(true));
        Path hostname = directory.resolve("hostname");
        Files.writeString(hostname, "#!/bin/sh\necho '10.1.2.3 10.1.2.4'\n");
        assertTrue(hostname.toFile().setExecutable(true));
        ProcessBuilder builder = new ProcessBuilder("sh", Path.of("whale/start.sh").toAbsolutePath().toString())
                .redirectErrorStream(true);
        var env = builder.environment();
        env.put("PATH", directory + ":" + env.get("PATH"));
        env.put("FLEXLB_MOCK_WHALE", "1");
        env.put("ROLE_TYPE", "PREFILL");
        env.put("START_PORT", "29750");
        env.put("FETCH_OUTPUT_STREAM", "1");
        env.put("MOCK_RUN_DIR", directory.toString());
        env.remove("POD_IP");
        env.remove("MOCK_PERFORMANCE_CONFIG");
        env.remove("MOCK_MASTER_CONFIG");
        String literal = "{\"label\":\"spaces $HOME 'quotes' `false`\"}";
        env.put("MOCK_PERFORMANCE_CONFIG_JSON", literal);
        env.put("MOCK_MASTER_CONFIG_JSON", "{}");
        Process process = builder.start();
        assertTrue(process.waitFor(3, TimeUnit.SECONDS));
        String args = new String(process.getInputStream().readAllBytes(), java.nio.charset.StandardCharsets.UTF_8);
        assertEquals(0, process.exitValue(), args);
        assertEquals(literal, Files.readString(directory.resolve("performance.json")));
        assertEquals("{}", Files.readString(directory.resolve("master.json")));
        assertTrue(args.contains("--host\n10.1.2.3\n"), args);
        assertTrue(args.contains("--auto-fetch\nfalse\n"), args);

        env.put("MOCK_PERFORMANCE_CONFIG", "ambiguous.json");
        Process refused = builder.start();
        assertTrue(refused.waitFor(3, TimeUnit.SECONDS));
        assertEquals(2, refused.exitValue());
    }

    private JavaMockEngineCluster.Config config(String... options) {
        List<String> args = new ArrayList<>(List.of("--endpoint-file", "endpoints.json",
                "--performance", "performance.json", "--master-config", "master.json"));
        args.addAll(List.of(options));
        return JavaMockEngineCluster.Config.parse(args.toArray(String[]::new));
    }

    @Test
    void defaultModeRetainsLocalDiscoveryAndNoPrivateMonitor() {
        var config = config();
        assertFalse(config.whale);
        assertFalse(config.kmonitor);
        assertNotNull(config.discoveryFile);
    }

    @Test
    void whaleRequiresOneEngineAndNeverGeneratesLocalDiscovery() {
        assertThrows(IllegalArgumentException.class, () -> config("--whale", "true"));
        var config = config("--whale", "true", "--n-prefill", "1", "--n-decode", "0",
                "--host", "10.1.2.3");
        assertNull(config.discoveryFile);
        assertEquals("10.1.2.3", JavaMockEngineCluster.declaredHost(config, 0));
        assertThrows(IllegalArgumentException.class, () -> config("--kmonitor", "true"));
    }

    @Test
    void internalProfileCreatesUsableMonitorAndOpenProfileFailsExplicitly() throws Exception {
        try {
            Class.forName("org.flexlb.monitor.FlexMonitorFactory");
        } catch (ClassNotFoundException absentInOpenProfile) {
            assertThrows(IllegalStateException.class, WhaleMockMonitor::create);
            return;
        }
        try (WhaleMockMonitor monitor = WhaleMockMonitor.create()) {
            assertNotNull(monitor);
            assertEquals("", Class.forName("com.taobao.kmonitor.impl.KMonitorConfig")
                    .getMethod("getKMonitorServiceName").invoke(null),
                    "engine series must not acquire the master whale-lb prefix");
        }
    }

    @Test
    void engineLabelsMatchExistingGrafanaWildcardFilters() {
        var tags = WhaleMockMonitor.engineTags(java.util.Map.of(
                "HIPPO_APP", "whale_prod_test", "HIPPO_ROLE", "test.prefill-cpu_part0",
                "HIPPO_SLAVE_IP", "10.0.0.1", "HIPPO_SERVICE_NAME", "test-group"), "10.1.0.2");
        assertEquals("whale_prod_test", tags.get("hippo_app"));
        assertEquals("test.prefill-cpu_part0", tags.get("hippo_role"));
        assertEquals("10.0.0.1", tags.get("host_ip"));
        assertEquals("10.1.0.2", tags.get("container_ip"));
        for (String key : List.of("dp_rank", "priority", "mtp_model_type", "pool")) {
            assertFalse(tags.get(key).isEmpty(), "wildcard filters require tag " + key);
        }
    }

    @Test
    void wallTpsUsesCounterDeltaAndReturnsToZeroWhenIdle() {
        java.util.Map<String, Double> values = new java.util.HashMap<>();
        var sink = (org.flexlb.metric.FlexMonitor) java.lang.reflect.Proxy.newProxyInstance(
                getClass().getClassLoader(), new Class<?>[]{org.flexlb.metric.FlexMonitor.class},
                (proxy, method, args) -> {
                    if (method.getName().equals("report") && args.length == 3)
                        values.put((String) args[0], ((Number) args[2]).doubleValue());
                    return null;
                });
        var monitor = new WhaleMockMonitor(sink);
        long now = System.nanoTime();
        monitor.sample(java.util.Map.of("mock_context_tokens_total", 0L), java.util.Map.of(), now);
        monitor.sample(java.util.Map.of("mock_context_tokens_total", 400L), java.util.Map.of(), now + 2_000_000_000L);
        assertEquals(200.0, values.get("rtp_llm_context_wall_tps_with_cache"));
        monitor.sample(java.util.Map.of("mock_context_tokens_total", 400L), java.util.Map.of(), now + 3_000_000_000L);
        assertEquals(0.0, values.get("rtp_llm_context_wall_tps_with_cache"));
    }

    @Test
    void wrapperRefusesImplicitActivationAndMapsWhalePortsAndFetch() throws Exception {
        Path script = Path.of("whale/start.sh").toAbsolutePath();
        ProcessBuilder refused = new ProcessBuilder("sh", script.toString()).redirectErrorStream(true);
        refused.environment().remove("FLEXLB_MOCK_WHALE");
        Process first = refused.start();
        assertTrue(first.waitFor(3, TimeUnit.SECONDS));
        assertEquals(2, first.exitValue());

        Path javaExecutable = directory.resolve("java");
        Files.writeString(javaExecutable, "#!/bin/sh\nprintf '%s\\n' \"$@\"\n");
        assertTrue(javaExecutable.toFile().setExecutable(true));
        ProcessBuilder builder = new ProcessBuilder("sh", script.toString()).redirectErrorStream(true);
        var env = builder.environment();
        env.put("PATH", directory + ":" + env.get("PATH"));
        env.put("FLEXLB_MOCK_WHALE", "1");
        env.put("ROLE_TYPE", "DECODE");
        env.put("START_PORT", "22290");
        env.put("POD_IP", "10.1.2.3");
        env.put("FETCH_OUTPUT_STREAM", "0");
        env.put("MOCK_PERFORMANCE_CONFIG", "performance.json");
        env.put("MOCK_MASTER_CONFIG", "master.json");
        env.put("MOCK_RUN_DIR", directory.toString());
        env.remove("MOCK_KMONITOR_ENABLED");
        Process process = builder.start();
        assertTrue(process.waitFor(3, TimeUnit.SECONDS));
        String args = new String(process.getInputStream().readAllBytes(), java.nio.charset.StandardCharsets.UTF_8);
        assertEquals(0, process.exitValue(), args);
        assertTrue(args.contains("--n-prefill\n0\n--n-decode\n1\n"), args);
        assertTrue(args.contains("--base-grpc-port\n22291\n--auto-fetch\ntrue\n"), args);
        assertTrue(args.contains("--kmonitor\ntrue\n"), args);
    }
}
