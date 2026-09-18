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
    void decodeSuccessQpsCountsOnlySuccessfulDecodeTerminals() {
        var reports = new ArrayList<Double>();
        var dashboardReports = new ArrayList<Double>();
        var dashboardTags = new ArrayList<java.util.Map<String, String>>();
        var sink = (org.flexlb.metric.FlexMonitor) java.lang.reflect.Proxy.newProxyInstance(
                getClass().getClassLoader(), new Class<?>[]{org.flexlb.metric.FlexMonitor.class},
                (proxy, method, args) -> {
                    if (method.getName().equals("report") && args.length == 3
                            && args[0].equals("mock_decode_success_qps"))
                        reports.add(((Number) args[2]).doubleValue());
                    if (method.getName().equals("report") && args.length == 3
                            && args[0].equals("py_rtp_success_qps_metric")) {
                        dashboardReports.add(((Number) args[2]).doubleValue());
                        dashboardTags.add(((org.flexlb.metric.FlexMetricTags) args[1]).getTags());
                    }
                    return null;
                });
        var monitor = new WhaleMockMonitor(sink);
        long now = System.nanoTime();
        var decode = java.util.Map.of("role", "ROLE_TYPE_DECODE", "engine", "decode-1",
                "hippo_role", "mock.decode_part0", "hippo_app", "mock-app");
        var prefill = java.util.Map.of("role", "ROLE_TYPE_PREFILL", "engine", "prefill-1");
        monitor.sample(java.util.Map.of("mock_completed_requests_total", 0), decode, now, true);
        monitor.sample(java.util.Map.of("mock_completed_requests_total", 8), decode, now + 2_000_000_000L, true);
        monitor.sample(java.util.Map.of("mock_completed_requests_total", 8), decode, now + 3_000_000_000L, true);
        monitor.sample(java.util.Map.of("mock_completed_requests_total", 20), prefill, now + 3_000_000_000L);
        assertEquals(List.of(0.0, 4.0, 0.0), reports);
        assertEquals(List.of(0.0, 8.0, 0.0), dashboardReports);
        assertEquals(java.util.Map.of("hippo_role", "mock.decode_part0", "hippo_app", "mock-app"),
                dashboardTags.get(1));
        monitor.sample(java.util.Map.of("mock_completed_requests_total", 9), decode, now + 4_000_000_000L);
        assertEquals(3, dashboardReports.size());
    }

    @Test
    void prefillFirstTokenLatencyUsesDashboardMillisecondsAndRoleOnly() {
        var values = new ArrayList<Double>();
        var labels = new ArrayList<java.util.Map<String, String>>();
        var sink = (org.flexlb.metric.FlexMonitor) java.lang.reflect.Proxy.newProxyInstance(
                getClass().getClassLoader(), new Class<?>[]{org.flexlb.metric.FlexMonitor.class},
                (proxy, method, args) -> {
                    if (method.getName().equals("report") && args[0].equals("py_rtp_response_first_token_rt")) {
                        values.add(((Number) args[2]).doubleValue());
                        labels.add(((org.flexlb.metric.FlexMetricTags) args[1]).getTags());
                    }
                    return null;
                });
        var monitor = new WhaleMockMonitor(sink);
        var prefill = java.util.Map.of("role", "ROLE_TYPE_PREFILL", "engine", "p-1",
                "hippo_role", "mock.prefill_part0", "hippo_app", "mock-app");
        monitor.reportEvent(java.util.Map.of("rtp_llm_first_token_latency_us", 415_000), prefill);
        monitor.reportEvent(java.util.Map.of("rtp_llm_first_token_latency_us", 42_000),
                java.util.Map.of("role", "ROLE_TYPE_DECODE", "hippo_role", "mock.decode_part0"));
        assertEquals(List.of(415.0), values);
        assertEquals(List.of(java.util.Map.of("hippo_role", "mock.prefill_part0", "hippo_app", "mock-app")),
                labels);
    }

    @Test
    void roleSpecificSamplesAndNoFetchDecodeLatencyStayOnTheirOwners() {
        var reports = new ArrayList<String>();
        var sink = (org.flexlb.metric.FlexMonitor) java.lang.reflect.Proxy.newProxyInstance(
                getClass().getClassLoader(), new Class<?>[]{org.flexlb.metric.FlexMonitor.class},
                (proxy, method, args) -> {
                    if (method.getName().equals("report"))
                        reports.add(args[0] + "=" + args[2]);
                    return null;
                });
        var monitor = new WhaleMockMonitor(sink);
        var p = java.util.Map.of("role", "ROLE_TYPE_PREFILL");
        var d = java.util.Map.of("role", "ROLE_TYPE_DECODE");
        long now = System.nanoTime();
        var counters = java.util.Map.<String, Number>of(
                "mock_context_compute_tokens_total", 100L,
                "mock_context_tokens_total", 100L,
                "mock_decode_step_tokens_total", 10L,
                "rtp_llm_context_batch_size", 1,
                "rtp_llm_generate_batch_size", 1);
        monitor.sample(counters, p, now);
        assertTrue(reports.stream().anyMatch(s -> s.startsWith("mock_context_tokens_total=")));
        assertFalse(reports.stream().anyMatch(s -> s.startsWith("rtp_llm_generate_tps=")));
        assertFalse(reports.stream().anyMatch(s -> s.startsWith("mock_decode_step_tokens_total=")));
        reports.clear();
        monitor.sample(counters, d, now);
        assertTrue(reports.stream().anyMatch(s -> s.startsWith("mock_decode_step_tokens_total=")));
        assertFalse(reports.stream().anyMatch(s -> s.startsWith("rtp_llm_context_tps=")));
        assertFalse(reports.stream().anyMatch(s -> s.startsWith("mock_context_tokens_total=")));
        reports.clear();
        monitor.reportEvent(java.util.Map.of("rtp_llm_latency_us", 425_000), d, true);
        assertTrue(reports.contains("rtp_llm_latency_us=425000.0"));
        assertTrue(reports.contains("py_rtp_framework_rt=425.0"));
        reports.clear();
        monitor.reportEvent(java.util.Map.of("rtp_llm_latency_us", 425_000), d, false);
        assertFalse(reports.stream().anyMatch(s -> s.startsWith("py_rtp_framework_rt=")));
        reports.clear();
        monitor.reportEvent(java.util.Map.of("rtp_llm_latency_us", 425_000), p, true);
        assertTrue(reports.isEmpty());
    }

    @Test
    void contextBatchSizeDoesNotMixIdlePollSamples() {
        var values = new ArrayList<Double>();
        var sink = (org.flexlb.metric.FlexMonitor) java.lang.reflect.Proxy.newProxyInstance(
                getClass().getClassLoader(), new Class<?>[]{org.flexlb.metric.FlexMonitor.class},
                (proxy, method, args) -> {
                    if (method.getName().equals("report") && args.length == 3
                            && args[0].equals("rtp_llm_context_batch_size"))
                        values.add(((Number) args[2]).doubleValue());
                    return null;
                });
        var monitor = new WhaleMockMonitor(sink);
        var tags = java.util.Map.<String, String>of();
        var idle = java.util.Map.<String, Number>of("rtp_llm_context_batch_size", 0);
        monitor.sample(idle, tags, System.nanoTime());
        monitor.reportEvent(java.util.Map.of("rtp_llm_context_batch_size", 1), tags);
        monitor.sample(idle, tags, System.nanoTime());
        monitor.sample(idle, tags, System.nanoTime());
        assertEquals(List.of(1.0), values);
    }

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
        assertBundleCompletion(false, false);
    }

    @Test
    @org.junit.jupiter.api.Timeout(15)
    void eosCompletesHugeOutputCapAndReleasesBothPoolsWithoutFetch() throws Exception {
        assertBundleCompletion(true, false);
    }

    @Test
    @org.junit.jupiter.api.Timeout(15)
    void bundleNonBatchStreamsTokensBeforeDecodeCompletesWithoutFetch() throws Exception {
        assertBundleCompletion(false, true);
    }

    private void assertBundleCompletion(boolean eos, boolean direct) throws Exception {
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
        if (direct) Files.writeString(perf, Files.readString(perf).replace("[[1,2]]", "[[1,30]]"));
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
            var decodeEvents = new java.util.concurrent.CopyOnWriteArrayList<java.util.Map<String, Number>>();
            reporter.set(d, (java.util.function.Consumer<java.util.Map<String, Number>>) decodeEvents::add);
            assertEquals("0", p.whaleMetricTags().get("dp_rank"));
            assertEquals("0", d.whaleMetricTags().get("dp_rank"));
            assertEquals(Integer.toString(port), p.whaleMetricTags().get("engine_port"));
            assertEquals(p.getHost(), p.whaleMetricTags().get("engine_ip"));
            assertNotEquals(p.whaleMetricTags().get("engine_port"), d.whaleMetricTags().get("engine_port"));
            channel = io.grpc.ManagedChannelBuilder.forAddress("127.0.0.1", port).usePlaintext().build();
            var input = org.flexlb.engine.grpc.EngineRpcService.GenerateInputPB.newBuilder()
                    .setRequestId(42).addAllTokenIds(java.util.Collections.nCopies(513, 123))
                    .setGenerateConfig(org.flexlb.engine.grpc.EngineRpcService.GenerateConfigPB.newBuilder()
                            .setMaxNewTokens(eos ? 393216 : 8).setMinNewTokens(8)
                            .setReuseCache(true).setEnableDeviceCache(true).setEnableMemoryCache(true)
                            .addRoleAddrs(org.flexlb.engine.grpc.EngineRpcService.RoleAddrPB.newBuilder()
                                    .setRole(org.flexlb.engine.grpc.EngineRpcService.RoleAddrPB.RoleType.DECODE)
                                    .setRoleStr("DECODE").setIp(d.getHost()).setGrpcPort(port + 1)));
            var batch = org.flexlb.engine.grpc.EngineRpcService.EnqueueBatchRequestPB.newBuilder()
                    .setBatchId(42).addDpSlots(org.flexlb.engine.grpc.EngineRpcService.EnqueueBatchDpSlotPB.newBuilder()
                            .setDpRank(0).addRequests(org.flexlb.engine.grpc.EngineRpcService.EnqueueBatchExternalInputPB.newBuilder()
                                    .setInput(input))).build();
            var stub = org.flexlb.engine.grpc.RpcServiceGrpc.newBlockingStub(channel)
                    .withDeadlineAfter(3, TimeUnit.SECONDS);
            if (direct) {
                var frames = stub.generateStreamCall(input.build());
                assertTrue(frames.hasNext());
                var first = frames.next().getFlattenOutput();
                assertFalse(first.getFinished(0));
                assertEquals(1, first.getAuxInfo(0).getOutputLen());
                assertEquals(4, first.getOutputIds().getInt32Data().size());
                assertEquals(0, d.getCompletedCount(), "P first frame must precede D completion");
                int tokens = 1;
                int count = 1;
                boolean terminal = false;
                while (frames.hasNext()) {
                    var frame = frames.next().getFlattenOutput();
                    assertFalse(terminal, "nothing follows the terminal frame");
                    tokens += frame.getOutputIds().getInt32Data().size() / Integer.BYTES;
                    terminal = frame.getFinished(0);
                    count++;
                }
                assertTrue(terminal);
                assertTrue(count > 2, "D progress must stream before terminal for TPOT");
                assertEquals(8, tokens, "first/progress/terminal must not duplicate tokens");
                assertEquals(0, ((Number) ((java.util.Map<?, ?>) p.getSnapshot().get("rpc_counts"))
                        .get("fetch_response")).intValue());
            } else {
                var ack = stub.enqueueBatch(batch);
                assertEquals(0, ack.getErrorsCount());
                assertEquals(1, ack.getSuccessesCount());
            }
            long deadline = System.nanoTime() + TimeUnit.SECONDS.toNanos(3);
            while ((d.getCompletedCount() != 1 || d.whaleMetrics().get("mock_generate_tokens_total").longValue() != 8)
                    && System.nanoTime() < deadline) Thread.sleep(10);
            assertEquals(1, d.getCompletedCount(), "D must complete without any Fetch RPC");
            assertEquals(0, d.getCancelledCount());
            assertEquals(1, ((Number) p.getSnapshot().get("cache_keys")).intValue());
            assertEquals(8, ((Number) d.getSnapshot().get("cache_keys")).intValue(),
                    "D must cache its eight 64-token blocks, not one P 512-token key");
            assertEquals(1, eventMetrics.stream().filter(m -> m.containsKey("rtp_llm_first_token_latency_us")).count());
            assertEquals(1, decodeEvents.stream().filter(m -> m.containsKey("rtp_llm_latency_us")).count(),
                    "D terminal reports inference residence without Fetch");
            var pForwards = eventMetrics.stream().filter(m -> m.containsKey("rtp_llm_model_forward_us"))
                    .map(m -> m.get("rtp_llm_model_forward_us").doubleValue()).toList();
            var dForwards = decodeEvents.stream().filter(m -> m.containsKey("rtp_llm_model_forward_us"))
                    .map(m -> m.get("rtp_llm_model_forward_us").doubleValue()).toList();
            assertEquals(List.of(2000.0), pForwards, "P reports one sample per batch, in microseconds");
            assertEquals(java.util.Collections.nCopies(8, direct ? 30000.0 : 2000.0), dForwards,
                    "D reports each forward step, not full request latency");

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
                "HIPPO_SLAVE_IP", "10.0.0.1", "HIPPO_SERVICE_NAME", "test-group"), "10.1.0.2", "ROLE_TYPE_PREFILL");
        assertEquals("whale_prod_test", tags.get("hippo_app"));
        assertEquals("test.prefill-cpu_part0", tags.get("hippo_role"));
        assertEquals("10.0.0.1", tags.get("host_ip"));
        assertEquals("10.1.0.2", tags.get("container_ip"));
        for (String key : List.of("dp_rank", "priority", "mtp_model_type", "pool")) {
            assertFalse(tags.get(key).isEmpty(), "wildcard filters require tag " + key);
        }
    }

    @Test
    void monitoringRoleAliasesAreExplicitAndDoNotRewritePlatformIdentity() {
        var env = java.util.Map.of("HIPPO_APP", "whale_prod_master_test",
                "HIPPO_ROLE", "master_test_deployment_g07.master_part",
                "MOCK_PREFILL_HIPPO_ROLE", "configured_p_alias",
                "MOCK_DECODE_HIPPO_ROLE", "configured_d_alias");
        var p = WhaleMockMonitor.engineTags(env, "10.1.0.2", "ROLE_TYPE_PREFILL");
        var d = WhaleMockMonitor.engineTags(env, "10.1.0.2", "ROLE_TYPE_DECODE");
        assertEquals("configured_p_alias", p.get("hippo_role"));
        assertEquals("configured_d_alias", d.get("hippo_role"));
        assertEquals("whale_prod_master_test", p.get("hippo_app"));
        assertEquals(p.get("hippo_app"), d.get("hippo_app"));
        assertEquals("master_test_deployment_g07.master_part", env.get("HIPPO_ROLE"));
        assertEquals(p.get("container_ip"), d.get("container_ip"));
        assertEquals(env.get("HIPPO_ROLE"), WhaleMockMonitor.engineTags(
                env, "10.1.0.2", "ROLE_TYPE_UNKNOWN").get("hippo_role"));
        var unset = java.util.Map.of("HIPPO_ROLE", "original.master_part");
        for (String role : List.of("ROLE_TYPE_PREFILL", "ROLE_TYPE_DECODE")) {
            assertEquals("original.master_part", WhaleMockMonitor.engineTags(unset, "10.1.0.2", role).get("hippo_role"));
        }
        assertEquals("original.master_part", WhaleMockMonitor.engineTags(java.util.Map.of(
                "HIPPO_ROLE", "original.master_part", "MOCK_PREFILL_HIPPO_ROLE", "  "),
                "10.1.0.2", "ROLE_TYPE_PREFILL").get("hippo_role"));
        assertEquals("", WhaleMockMonitor.engineTags(
                java.util.Map.of(), "10.1.0.2", "ROLE_TYPE_PREFILL").get("hippo_role"));
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
    void prefillRunningReportsCompletionZeroButDoesNotRepeatIdleSamples() {
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
        var tags = java.util.Map.of("role", "ROLE_TYPE_PREFILL");
        var idle = java.util.Map.<String, Number>of("rtp_llm_running_stream_size", 0);
        long now = System.nanoTime();
        monitor.sample(idle, tags, now);
        monitor.reportScheduler(java.util.Map.of("rtp_llm_running_stream_size", 1), tags);
        monitor.reportScheduler(idle, tags);
        for (int i = 1; i <= 10; i++)
            monitor.sample(idle, tags, now + i * 5_000_000_000L);
        assertEquals(List.of(1.0, 0.0), running);
        // Decode keeps its existing periodic idle reporting.
        monitor.sample(idle, java.util.Map.of("role", "ROLE_TYPE_DECODE"), now);
        assertEquals(List.of(1.0, 0.0, 0.0), running);
    }

    @Test
    void schedulerSamplesSuppressOnlyTheirPollingInterval() {
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
