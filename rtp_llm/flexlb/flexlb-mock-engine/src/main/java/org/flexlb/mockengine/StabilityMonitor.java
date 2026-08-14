package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ArrayNode;
import com.fasterxml.jackson.databind.node.ObjectNode;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.time.Instant;
import java.time.ZoneOffset;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Standalone stability monitor that replaces {@code stability_monitor.py}.
 *
 * <p>Polls FlexLB master inflight status and mock engine snapshots at a fixed
 * interval, writing one JSON line per poll to {@code monitor.jsonl}. Detects
 * inflight leaks by tracking per-engine completion progress: if an engine's
 * completed count stays unchanged while its inflight remains positive beyond a
 * grace period, a leak is flagged.
 *
 * <p>Configuration is entirely via environment variables (single source of truth):
 * <ul>
 *   <li>{@code FLEXLB_HTTP_ADDR} — FlexLB master HTTP address (e.g. "127.0.0.1:7002")</li>
 *   <li>{@code MOCK_CONTROL_HOST} — Mock engine control host</li>
 *   <li>{@code MOCK_CONTROL_PORT} — Mock engine control port</li>
 *   <li>{@code OUTPUT_DIR} — Output directory for monitor.jsonl</li>
 *   <li>{@code POLL_INTERVAL_S} — Poll interval in seconds (default 5)</li>
 *   <li>{@code LEAK_GRACE_S} — Leak detection grace period in seconds (default 60)</li>
 * </ul>
 *
 * <p>Run with: {@code java -cp <jar> org.flexlb.mockengine.StabilityMonitor}
 */
public final class StabilityMonitor {

    private static final ObjectMapper OBJECT_MAPPER = new ObjectMapper();
    private static final DateTimeFormatter ISO_FORMATTER =
            DateTimeFormatter.ISO_INSTANT.withZone(ZoneOffset.UTC);
    private static final Duration HTTP_TIMEOUT = Duration.ofSeconds(3);

    private StabilityMonitor() {
    }

    public static void main(String[] args) throws Exception {
        Config config = Config.parse();

        HttpClient httpClient = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(2))
                .build();

        Path outputPath = Path.of(config.outputDir).resolve("monitor.jsonl");
        Files.createDirectories(outputPath.toAbsolutePath().getParent());

        AtomicBoolean running = new AtomicBoolean(true);
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            running.set(false);
            System.err.println("[monitor] shutdown signal received");
        }, "stability-monitor-shutdown"));

        // Per-engine leak tracking: port -> [lastCompleted, lastChangeEpochMs]
        Map<Integer, long[]> leakState = new HashMap<>();
        long leakGraceMs = config.leakGraceS * 1000L;

        int pollCount = 0;
        boolean leakEverDetected = false;
        boolean mockWarningLogged = false;

        try (var writer = Files.newBufferedWriter(outputPath)) {
            System.err.println("[monitor] writing to " + outputPath);
            System.err.printf("[monitor] flexlb=%s mock=%s:%d interval=%ds grace=%ds%n",
                    config.flexlbHttpAddr, config.mockControlHost,
                    config.mockControlPort, config.pollIntervalS, config.leakGraceS);

            while (running.get()) {
                long pollStart = System.currentTimeMillis();
                Instant now = Instant.ofEpochMilli(pollStart);

                // 1. Poll FlexLB inflight status
                InflightData inflight = pollInflight(httpClient, config);

                // 2. Poll mock engine snapshot
                List<MockEngineData> engines = pollMockSnapshot(httpClient, config);
                if (engines == null) {
                    if (!mockWarningLogged) {
                        System.err.println("[monitor] WARNING: mock engine /snapshot not available, "
                                + "skipping mock polling");
                        mockWarningLogged = true;
                    }
                } else if (mockWarningLogged) {
                    System.err.println("[monitor] mock engine /snapshot recovered");
                    mockWarningLogged = false;
                }

                // 3. Build JSONL record
                ObjectNode record = OBJECT_MAPPER.createObjectNode();
                record.put("timestamp", ISO_FORMATTER.format(now));

                int schedulerInflight = inflight != null ? inflight.schedulerInflight() : 0;
                record.put("scheduler_inflight", schedulerInflight);

                ArrayNode prefillArr = record.putArray("prefill_inflight");
                if (inflight != null) {
                    for (int v : inflight.prefillInflight()) {
                        prefillArr.add(v);
                    }
                }

                ArrayNode decodeArr = record.putArray("decode_inflight");
                if (inflight != null) {
                    for (int v : inflight.decodeInflight()) {
                        decodeArr.add(v);
                    }
                }

                // 4. Mock engines + per-engine leak detection
                int mockTotalInflight = 0;
                boolean globalLeak = false;
                ArrayNode enginesArr = record.putArray("mock_engines");

                if (engines != null) {
                    for (MockEngineData engine : engines) {
                        int engineInflight = engine.inflight();
                        mockTotalInflight += engineInflight;

                        // Per-engine leak detection: completed unchanged while inflight > 0
                        long[] state = leakState.computeIfAbsent(
                                engine.port(), k -> new long[]{engine.completed(), pollStart});
                        boolean engineLeak = false;
                        if (engineInflight > 0) {
                            if (engine.completed() != state[0]) {
                                state[0] = engine.completed();
                                state[1] = pollStart;
                            } else if (pollStart - state[1] >= leakGraceMs) {
                                engineLeak = true;
                                globalLeak = true;
                                System.err.printf("[monitor] WARN: leak on engine port=%d "
                                                + "inflight=%d completed=%d (no completion for %dms)%n",
                                        engine.port(), engineInflight, engine.completed(),
                                        pollStart - state[1]);
                            }
                        } else {
                            // Inflight drained — reset so a future stall starts fresh
                            state[0] = engine.completed();
                            state[1] = pollStart;
                        }

                        ObjectNode engineNode = enginesArr.addObject();
                        engineNode.put("port", engine.port());
                        engineNode.put("running", engine.running());
                        engineNode.put("accepted", engine.accepted());
                        engineNode.put("completed", engine.completed());
                        engineNode.put("inflight", engineInflight);
                        engineNode.put("leak_detected", engineLeak);
                    }
                }

                // total_inflight: prefer scheduler view, fall back to mock sum if master is down
                int totalInflight = schedulerInflight;
                if (inflight == null) {
                    totalInflight = mockTotalInflight;
                }
                record.put("total_inflight", totalInflight);
                record.put("leak_detected", globalLeak);
                if (globalLeak) {
                    leakEverDetected = true;
                }

                writer.write(OBJECT_MAPPER.writeValueAsString(record));
                writer.newLine();
                writer.flush();
                pollCount++;

                // 5. Sleep for remaining interval (1-second chunks for responsive shutdown)
                long elapsed = System.currentTimeMillis() - pollStart;
                long sleepMs = Math.max(0, config.pollIntervalS * 1000L - elapsed);
                while (sleepMs > 0 && running.get()) {
                    long chunk = Math.min(sleepMs, 1000);
                    Thread.sleep(chunk);
                    sleepMs -= chunk;
                }
            }
        }

        System.err.printf("[monitor] stopped after %d polls, wrote %s%n", pollCount, outputPath);
        System.err.printf("[monitor] leak ever detected: %s%n", leakEverDetected);
    }

    // ---- HTTP polling ----

    private static InflightData pollInflight(HttpClient client, Config config) {
        String url = "http://" + config.flexlbHttpAddr + "/rtp_llm/inflight_status";
        String body = httpGet(client, url);
        if (body == null) {
            return null;
        }
        try {
            JsonNode root = OBJECT_MAPPER.readTree(body);
            int schedulerInflight = root.path("scheduler_inflight").asInt(0);

            List<Integer> prefill = new ArrayList<>();
            for (JsonNode ep : root.path("prefill_endpoints")) {
                prefill.add(ep.path("inflight_batches").asInt(0));
            }

            List<Integer> decode = new ArrayList<>();
            for (JsonNode ep : root.path("decode_endpoints")) {
                decode.add(ep.path("inflight_requests").asInt(0));
            }

            return new InflightData(schedulerInflight, prefill, decode);
        } catch (Exception e) {
            System.err.printf("[monitor] failed to parse inflight_status: %s%n", e.getMessage());
            return null;
        }
    }

    private static List<MockEngineData> pollMockSnapshot(HttpClient client, Config config) {
        String url = "http://" + config.mockControlHost + ":" + config.mockControlPort + "/snapshot";
        String body = httpGet(client, url);
        if (body == null) {
            return null;
        }
        try {
            JsonNode root = OBJECT_MAPPER.readTree(body);
            // MockControlServer returns a JSON array directly as the root element.
            // For backward compatibility, also handle {"engines":[...]} wrapper.
            JsonNode enginesNode;
            if (root.isArray()) {
                enginesNode = root;
            } else {
                enginesNode = root.path("engines");
            }
            if (!enginesNode.isArray()) {
                return null;
            }
            List<MockEngineData> engines = new ArrayList<>(enginesNode.size());
            for (JsonNode e : enginesNode) {
                int port = e.has("port") ? e.get("port").asInt()
                        : e.path("grpc_port").asInt(0);
                int running = e.path("running").asInt(0);
                long accepted = e.path("accepted").asLong(0);
                long completed = e.path("completed").asLong(0);
                int inflight = e.has("inflight") ? e.get("inflight").asInt()
                        : (int) Math.max(0, accepted - completed);
                engines.add(new MockEngineData(port, running, accepted, completed, inflight));
            }
            return engines;
        } catch (Exception e) {
            System.err.printf("[monitor] failed to parse mock snapshot: %s%n", e.getMessage());
            return null;
        }
    }

    private static String httpGet(HttpClient client, String url) {
        try {
            HttpRequest request = HttpRequest.newBuilder()
                    .uri(URI.create(url))
                    .timeout(HTTP_TIMEOUT)
                    .GET()
                    .build();
            HttpResponse<String> response = client.send(request, HttpResponse.BodyHandlers.ofString());
            if (response.statusCode() != 200) {
                return null;
            }
            return response.body();
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return null;
        } catch (Exception e) {
            return null;
        }
    }

    // ---- Inner types ----

    private record InflightData(int schedulerInflight, List<Integer> prefillInflight,
                                List<Integer> decodeInflight) {
    }

    private record MockEngineData(int port, int running, long accepted, long completed,
                                  int inflight) {
    }

    private static final class Config {
        private String flexlbHttpAddr;
        private String mockControlHost;
        private int mockControlPort;
        private String outputDir;
        private int pollIntervalS = 5;
        private int leakGraceS = 60;

        private static Config parse() {
            Config config = new Config();
            config.flexlbHttpAddr = System.getenv("FLEXLB_HTTP_ADDR");
            config.mockControlHost = System.getenv("MOCK_CONTROL_HOST");
            config.outputDir = System.getenv("OUTPUT_DIR");

            String mockPort = System.getenv("MOCK_CONTROL_PORT");
            String interval = System.getenv("POLL_INTERVAL_S");
            String grace = System.getenv("LEAK_GRACE_S");

            if (mockPort != null && !mockPort.isEmpty()) {
                config.mockControlPort = Integer.parseInt(mockPort);
            }
            if (interval != null && !interval.isEmpty()) {
                config.pollIntervalS = Integer.parseInt(interval);
            }
            if (grace != null && !grace.isEmpty()) {
                config.leakGraceS = Integer.parseInt(grace);
            }

            if (config.flexlbHttpAddr == null || config.flexlbHttpAddr.isEmpty()) {
                throw new IllegalArgumentException("FLEXLB_HTTP_ADDR is required");
            }
            if (config.mockControlHost == null || config.mockControlHost.isEmpty()) {
                throw new IllegalArgumentException("MOCK_CONTROL_HOST is required");
            }
            if (config.mockControlPort == 0) {
                throw new IllegalArgumentException("MOCK_CONTROL_PORT is required");
            }
            if (config.outputDir == null || config.outputDir.isEmpty()) {
                throw new IllegalArgumentException("OUTPUT_DIR is required");
            }
            return config;
        }
    }
}
