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
    void bundleIsExplicitAndRetainsFileDiscoveryForAllLogicalEngines() {
        assertThrows(IllegalArgumentException.class, () -> config("--whale-bundle", "true"));
        var bundle = config("--whale", "true", "--whale-bundle", "true",
                "--host", "10.1.2.3", "--n-prefill", "48", "--n-decode", "192",
                "--auto-fetch", "true", "--kmonitor", "true");
        assertNotNull(bundle.discoveryFile);
        assertTrue(bundle.autoFetch);
        assertEquals(240, bundle.nPrefill + bundle.nDecode);
        assertEquals(JavaMockEngineCluster.derivedLoopbackIp(239), JavaMockEngineCluster.declaredHost(bundle, 239));
        assertNotEquals(JavaMockEngineCluster.declaredHost(bundle, 0), JavaMockEngineCluster.declaredHost(bundle, 239));
    }

    @Test
    @org.junit.jupiter.api.Timeout(15)
    void bundleCompletesWithoutFetchWithOneSharedNetworkWorker() throws Exception {
        assertBundleCompletion(false);
    }

    @Test
    @org.junit.jupiter.api.Timeout(15)
    void eosCompletesHugeOutputCapAndReleasesBothPoolsWithoutFetch() throws Exception {
        assertBundleCompletion(true);
    }

    private void assertBundleCompletion(boolean eos) throws Exception {
        var cfg = config("--whale", "true", "--whale-bundle", "true", "--auto-fetch", "true",
                "--host", "10.1.2.3", "--n-prefill", "1", "--n-decode", "1",
                "--prefill-block-size", "512", "--decode-block-size", "64",
                "--prefill-kv-pool-blocks", "32", "--decode-kv-pool-blocks", "32");
        Path perf = directory.resolve("bundle-perf.json");
        Path master = directory.resolve("bundle-master.json");
        Files.writeString(perf, "{\"block_size\":1024,\"sleep_scale\":1,\"jitter_pct\":0,"
                + "\"prefill\":{\"scale\":1},\"decode\":{\"scale\":1,\"tokens_per_step\":1,"
                + "\"step_ms_by_batch\":[[1,2]]}}");
        if (eos) {
            var mapper = new com.fasterxml.jackson.databind.ObjectMapper();
            var tree = mapper.readTree(perf.toFile());
            ((com.fasterxml.jackson.databind.node.ObjectNode) tree.get("decode")).set("eos",
                    mapper.readTree("{\"enabled\":true,\"distribution\":\"geometric\",\"mean_tokens\":1,\"seed\":42}"));
            mapper.writeValue(perf.toFile(), tree);
        }
        MockMasterConfig.writeWithPrefillExpression(master, "2");
        var model = MockPerformanceModel.load(perf.toString(), master.toString());
        var boss = new io.netty.channel.nio.NioEventLoopGroup(1);
        var worker = new io.netty.channel.nio.NioEventLoopGroup(1);
        var scheduler = java.util.concurrent.Executors.newScheduledThreadPool(2);
        var services = new java.util.concurrent.ConcurrentHashMap<Integer, JavaMockEngineCluster.FastRpcService>();
        var servers = new java.util.concurrent.ConcurrentHashMap<Integer, io.grpc.Server>();
        io.grpc.ManagedChannel channel = null;
        int port = Integer.parseInt(System.getenv().getOrDefault("FLEXLB_PORT_BASE", "62600")) + 10;
        try {
            var stats = new JavaMockEngineCluster.ClusterStats();
            var p = JavaMockEngineCluster.startEngine(cfg, model, servers, boss, worker, services,
                    scheduler, stats, "prefill", "prefill-0", port, 0);
            var d = JavaMockEngineCluster.startEngine(cfg, model, servers, boss, worker, services,
                    scheduler, stats, "decode", "decode-0", port + 1, 1);
            assertEquals(32L * 512, p.getTotalKvTokens());
            assertEquals(32L * 64, d.getTotalKvTokens());
            assertEquals(1024, model.blockSize(), "role overrides must not mutate the shared model");
            var eventMetrics = new java.util.concurrent.CopyOnWriteArrayList<java.util.Map<String, Number>>();
            var reporter = p.getClass().getDeclaredField("eventMetricReporter");
            reporter.setAccessible(true);
            reporter.set(p, (java.util.function.Consumer<java.util.Map<String, Number>>) eventMetrics::add);
            channel = io.grpc.ManagedChannelBuilder.forAddress("127.0.0.1", port).usePlaintext().build();
            var input = org.flexlb.engine.grpc.EngineRpcService.GenerateInputPB.newBuilder()
                    .setRequestId(42).addTokenIds(123)
                    .setGenerateConfig(org.flexlb.engine.grpc.EngineRpcService.GenerateConfigPB.newBuilder()
                            .setMaxNewTokens(eos ? 393216 : 8).setMinNewTokens(8)
                            .addRoleAddrs(org.flexlb.engine.grpc.EngineRpcService.RoleAddrPB.newBuilder()
                                    .setRole(org.flexlb.engine.grpc.EngineRpcService.RoleAddrPB.RoleType.DECODE)
                                    .setRoleStr("DECODE").setIp(d.getHost()).setGrpcPort(port + 1)));
            var batch = org.flexlb.engine.grpc.EngineRpcService.EnqueueBatchRequestPB.newBuilder()
                    .setBatchId(42).addDpSlots(org.flexlb.engine.grpc.EngineRpcService.EnqueueBatchDpSlotPB.newBuilder()
                            .setDpRank(0).addRequests(org.flexlb.engine.grpc.EngineRpcService.EnqueueBatchExternalInputPB.newBuilder()
                                    .setInput(input))).build();
            var ack = org.flexlb.engine.grpc.RpcServiceGrpc.newBlockingStub(channel)
                    .withDeadlineAfter(2, TimeUnit.SECONDS).enqueueBatch(batch);
            assertEquals(0, ack.getErrorsCount());
            assertEquals(1, ack.getSuccessesCount());
            long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(3);
            while ((d.getCompletedCount() != 1 || d.whaleMetrics().get("mock_generate_tokens_total").longValue() != 8)
                    && System.nanoTime() < deadline) Thread.sleep(10);
            assertEquals(1, d.getCompletedCount(), "D must complete without any Fetch RPC");
            assertEquals(0, d.getCancelledCount());
            assertEquals(1, eventMetrics.stream().filter(m -> m.containsKey("rtp_llm_first_token_latency_us")).count());

            assertEquals(0, d.getRunningCount());
            assertEquals(8, d.whaleMetrics().get("mock_generate_tokens_total").longValue());
            long releaseDeadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(2);
            while ((p.getActiveKvTokens() != 0 || d.getActiveKvTokens() != 0)
                    && System.nanoTime() < releaseDeadline) Thread.sleep(10);
            assertEquals(0, p.getActiveKvTokens(), "P connector lease released");
            assertEquals(0, d.getActiveKvTokens(), "EOS releases D KV through normal completion");
            assertEquals(0, d.getInflightCount());
            assertTrue(p.whaleMetrics().get("mock_context_tokens_total").longValue() > 0);
            assertFalse(p.isWhaleRemote());
            assertFalse(d.isWhaleRemote());
        } finally {
            if (channel != null) channel.shutdownNow();
            for (var server : servers.values()) server.shutdownNow();
            for (var service : services.values()) service.shutdown();
            scheduler.shutdownNow();
            boss.shutdownGracefully(0, 1, TimeUnit.SECONDS).syncUninterruptibly();
            worker.shutdownGracefully(0, 1, TimeUnit.SECONDS).syncUninterruptibly();
        }
    }

    @Test
    void bundledEnginesNeverShareCounterDeltaOrSamplingClock() {
        List<Double> rates = new ArrayList<>();
        var sink = (org.flexlb.metric.FlexMonitor) java.lang.reflect.Proxy.newProxyInstance(
                getClass().getClassLoader(), new Class<?>[]{org.flexlb.metric.FlexMonitor.class},
                (proxy, method, args) -> {
                    if (method.getName().equals("report") && args.length == 3
                            && args[0].equals("rtp_llm_generate_tps"))
                        rates.add(((Number) args[2]).doubleValue());
                    return null;
                });
        var monitor = new WhaleMockMonitor(sink);
        var a = java.util.Map.of("engine", "decode-0");
        var b = java.util.Map.of("engine", "decode-1");
        long now = System.nanoTime();
        monitor.sample(java.util.Map.of("mock_decode_step_tokens_total", 100L), a, now);
        monitor.sample(java.util.Map.of("mock_decode_step_tokens_total", 800L), b, now);
        rates.clear();
        monitor.sample(java.util.Map.of("mock_decode_step_tokens_total", 300L), a, now + 2_000_000_000L);
        monitor.sample(java.util.Map.of("mock_decode_step_tokens_total", 1400L), b, now + 2_000_000_000L);
        assertEquals(List.of(100.0, 300.0), rates);
        monitor.sample(java.util.Map.of("mock_decode_step_tokens_total", 300L), a, now + 3_000_000_000L);
        assertEquals(0.0, rates.get(2));
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
        monitor.sample(java.util.Map.of("mock_context_tokens_total", 0L, "mock_context_with_cache_ms_total", 0L), java.util.Map.of(), now);
        monitor.sample(java.util.Map.of("mock_context_tokens_total", 400L, "mock_context_with_cache_ms_total", 10L), java.util.Map.of(), now + 2_000_000_000L);
        assertEquals(200.0, values.get("rtp_llm_context_wall_tps_with_cache"));
        assertEquals(40000.0, values.get("rtp_llm_context_tps_with_cache"));
        monitor.sample(java.util.Map.of("mock_context_tokens_total", 400L, "mock_context_with_cache_ms_total", 10L), java.util.Map.of(), now + 3_000_000_000L);
        assertEquals(0.0, values.get("rtp_llm_context_wall_tps_with_cache"));
        assertEquals(0.0, values.get("rtp_llm_context_tps_with_cache"));
    }

    @Test
    void shortPrefillBetweenPollsIsReportedAndIdleReturnsToZero() {
        List<Double> running = new ArrayList<>();
        var sink = (org.flexlb.metric.FlexMonitor) java.lang.reflect.Proxy.newProxyInstance(
                getClass().getClassLoader(), new Class<?>[]{org.flexlb.metric.FlexMonitor.class},
                (proxy, method, args) -> {
                    if (method.getName().equals("report") && args.length == 3
                            && args[0].equals("rtp_llm_running_stream_size"))
                        running.add(((Number) args[2]).doubleValue());
                    return null;
                });
        var monitor = new WhaleMockMonitor(sink);
        long now = System.nanoTime();
        var idle = java.util.Map.<String, Number>of("rtp_llm_running_stream_size", 0);
        monitor.sample(idle, java.util.Map.of(), now);
        // Both batches finish before the next five-second polling boundary.
        monitor.reportScheduler(java.util.Map.of("rtp_llm_running_stream_size", 3), java.util.Map.of());
        monitor.reportScheduler(java.util.Map.of("rtp_llm_running_stream_size", 1), java.util.Map.of());
        monitor.sample(idle, java.util.Map.of(), now + 5_000_000_000L);
        assertEquals(List.of(0.0, 3.0, 1.0), running);
        monitor.sample(idle, java.util.Map.of(), now + 10_000_000_000L);
        assertEquals(List.of(0.0, 3.0, 1.0, 0.0), running);
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
