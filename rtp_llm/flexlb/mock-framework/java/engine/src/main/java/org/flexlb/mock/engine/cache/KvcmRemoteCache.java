package org.flexlb.mock.engine.cache;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import java.io.IOException;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import java.util.Map;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.List;
import java.util.UUID;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.Semaphore;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import org.flexlb.mock.engine.config.RtpConfig;
import org.flexlb.mock.engine.scheduler.EngineScheduler.Input;
import org.flexlb.mock.engine.scheduler.EngineScheduler.RemotePrefix;
import org.flexlb.mock.metrics.EventMetrics;

/**
 * Shares metadata-only Remote Cache through a stable deployment identity and dummy storage.
 * Registers the instance before reading or publishing; unavailable KVCM falls back to local compute.
 */
public final class KvcmRemoteCache implements AutoCloseable {
    private static final System.Logger LOG = System.getLogger(KvcmRemoteCache.class.getName());
    private static final ObjectMapper JSON = new ObjectMapper();
    private static final Duration WRITE_TIMEOUT = Duration.ofMillis(1500);

    private final ObjectNode registration;
    private final String instance;

    private volatile URI registeredEndpoint;

    private final Duration readTimeout;
    private final KvcmDiscovery endpoint;
    private final KvCacheManager local;
    private final EventMetrics metrics;
    private final HttpClient http;
    private final Semaphore reads = new Semaphore(32);
    private final ThreadPoolExecutor writer = new ThreadPoolExecutor(1, 1, 0, TimeUnit.SECONDS,
        new ArrayBlockingQueue<>(128), Thread.ofPlatform().daemon().name("mock-remote-writer").factory());

    private KvcmRemoteCache(ObjectNode registration, Duration readTimeout, KvcmDiscovery endpoint,
                            KvCacheManager local, EventMetrics metrics) {
        this.registration = registration;
        this.instance = registration.path("instance_id").asText();
        this.readTimeout = readTimeout;
        this.endpoint = endpoint;
        this.local = local;
        this.metrics = metrics;
        http = HttpClient.newBuilder().connectTimeout(readTimeout).build();
        writer.execute(this::discover);
    }

    public static KvcmRemoteCache start(RtpConfig config, KvCacheManager local, EventMetrics metrics) {
        if (!config.remoteCache()) return null;
        var values = config.business();
        var registration = registration(config, System.getenv());
        String instance = registration.path("instance_id").asText();
        String address = values.path("mock.cache.remote.endpoint").asText();
        boolean vip = RtpConfig.nativeBool(values, "kvcm_enable_vipserver", false);
        var timeout = Duration.ofMillis(RtpConfig.nativeInt(values, "mock.cache.remote.read.timeout.ms", 20, 1, 1000));
        var endpoint = address.isBlank()
            ? KvcmDiscovery.discover(values.path(vip ? "kvcm_vipserver_domain" : "kvcm_server_address").asText(), instance, vip)
            : KvcmDiscovery.direct(URI.create(address));
        try {
            return new KvcmRemoteCache(registration, timeout, endpoint, local, metrics);
        } catch (RuntimeException | Error error) {
            endpoint.close();
            throw error;
        }
    }

    /** Run on the request worker before Engine admission. Never resolve service discovery on this path. */
    public Input lookup(Input input) {
        if (!input.reuse()) return input;
        int limit = Math.min(input.keys().size(), (input.tokens() - 1) / input.promptCacheKeys().tokensPerKey());
        var keys = input.keys().subList(0, limit);
        int start = local.localPrefixLength(keys);
        if (start == limit) return input;
        URI target = endpoint.current();
        if (target == null || !target.equals(registeredEndpoint) || !reads.tryAcquire()) {
            metrics.increment("mock_remote_cache_read_skipped");
            return input;
        }
        long began = System.nanoTime();
        try {
            long budget = input.deadline() == Long.MAX_VALUE ? readTimeout.toNanos()
                : Math.min(readTimeout.toNanos(), input.deadline() - began);
            if (budget <= 0) return input;
            var request = request(keys.subList(start, limit)).put("query_type", "QT_PREFIX_MATCH");
            var result = post(target, "/api/getCacheLocationLen", request, Duration.ofNanos(budget));
            var count = result.has("cache_location_len") ? result.get("cache_location_len") : result.get("cacheLocationLen");
            int hits = count == null ? 0 : Integer.parseInt(count.asText());
            if (hits < 0 || hits > limit - start) throw new IOException("Invalid KVCM prefix length");
            metrics.increment("mock_remote_cache_read_success");
            return input.withRemotePrefix(new RemotePrefix(start, start + hits));
        } catch (InterruptedException error) {
            Thread.currentThread().interrupt();
            return input;
        } catch (IOException | RuntimeException error) {
            metrics.increment("mock_remote_cache_read_failure");
            return input;
        } finally {
            metrics.observe("mock_remote_cache_read_us", (System.nanoTime() - began) / 1000.0);
            reads.release();
        }
    }

    /** Publish only after successful Prefill completion. The Engine notification thread never waits for KVCM. */
    public void publish(Input input) {
        if (!input.reuse() || input.keys().isEmpty()) return;
        try {
            writer.execute(() -> write(input.keys()));
        } catch (RejectedExecutionException error) {
            metrics.increment("mock_remote_cache_write_dropped");
        }
    }

    @Override public void close() {
        writer.shutdownNow();
        endpoint.close();
        http.shutdownNow();
    }

    // All P replicas with the same deployment and cache layout share one instance, including after restart.
    static ObjectNode registration(RtpConfig config, Map<String, String> env) {
        var values = config.business();
        String group = values.path("kvcm_instance_group").asText("default");
        if (group.isBlank()) throw new IllegalArgumentException("Empty kvcm_instance_group");
        var identity = JSON.createObjectNode().put("format", 1).put("instance_group", group)
            .put("salt", values.path("kvcm_instance_id_salt").asText())
            .put("biz", env.getOrDefault("BIZ_NAME", ""))
            .put("deployment", env.getOrDefault("DEPLOYMENT_NAME", ""))
            .put("model", values.path("model_name").asText("mock"))
            .put("block_size", config.tokensPerCacheKey()).put("cp_size", config.cpSize())
            .put("cache_key_hash_seed", config.cacheKeySeed()).put("image_token_id", config.imageToken());
        String instance;
        try {
            instance = "mock_remote_" + HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256")
                .digest(identity.toString().getBytes(StandardCharsets.UTF_8)));
        } catch (NoSuchAlgorithmException error) {
            throw new IllegalStateException("SHA-256 unavailable", error);
        }
        var result = JSON.createObjectNode().put("instance_group", group).put("instance_id", instance)
            .put("block_size", config.tokensPerCacheKey()).put("default_query_type", "QT_PREFIX_MATCH");
        result.putObject("model_deployment").put("model_name", values.path("model_name").asText("mock"))
            .put("dtype", "fp8").put("use_mla", false).put("tp_size", 1).put("dp_size", 1).put("pp_size", 1);
        // Dummy storage tracks metadata only; 4096 is its logical per-block accounting unit, not real KV bytes.
        result.putArray("location_spec_infos").addObject().put("name", "mock_kv").put("size", "4096");
        return result;
    }

    private void discover() {
        try {
            ensureRegistered(endpoint.get());
        } catch (IOException | RuntimeException error) {
            metrics.increment("mock_remote_cache_discovery_failure");
        } catch (InterruptedException error) {
            Thread.currentThread().interrupt();
        }
    }

    private void ensureRegistered(URI target) throws IOException, InterruptedException {
        if (target.equals(registeredEndpoint)) return;
        post(target, "/api/registerInstance", registration.deepCopy().put("trace_id", UUID.randomUUID().toString()), WRITE_TIMEOUT);
        registeredEndpoint = target;
        LOG.log(System.Logger.Level.INFO, "Remote cache registered: instance=" + instance
            + " group=" + registration.path("instance_group").asText());
        metrics.increment("mock_remote_cache_registration_success");
    }

    private JsonNode post(URI target, String path, ObjectNode request, Duration timeout)
            throws IOException, InterruptedException {
        var call = http.sendAsync(HttpRequest.newBuilder(target.resolve(path)).timeout(timeout)
            .header("Content-Type", "application/json").POST(HttpRequest.BodyPublishers.ofString(request.toString()))
            .build(), HttpResponse.BodyHandlers.ofString());
        HttpResponse<String> response;
        try {
            response = call.get(timeout.toNanos(), TimeUnit.NANOSECONDS);
        } catch (TimeoutException | InterruptedException error) {
            call.cancel(true);
            if (error instanceof InterruptedException interrupted) throw interrupted;
            throw new IOException("KVCM request timed out", error);
        } catch (ExecutionException error) {
            if (error.getCause() instanceof java.net.ConnectException) endpoint.invalidate(target);
            throw new IOException("KVCM request failed", error.getCause());
        }
        if (response.statusCode() != 200) throw new IOException("KVCM HTTP status " + response.statusCode());
        var body = JSON.readTree(response.body());
        if (body == null) throw new IOException("Empty KVCM response");
        String code = body.path("header").path("status").path("code").asText();
        if (java.util.Set.of("8", "9", "10", "INSTANCE_NOT_EXIST", "SERVER_NOT_LEADER", "NODE_NOT_REGISTERED").contains(code))
            registeredEndpoint = null;
        if (code.equals("9") || code.equals("SERVER_NOT_LEADER")) endpoint.invalidate(target);
        if (!code.equals("1") && !code.equals("OK")) throw new IOException("KVCM status " + code);
        return body;
    }

    private void write(List<Long> keys) {
        try {
            URI target = endpoint.get();
            ensureRegistered(target);
            var start = post(target, "/api/startWriteCache", request(keys).put("write_timeout_seconds", 5), WRITE_TIMEOUT);
            String session = start.path("write_session_id").asText(start.path("writeSessionId").asText());
            if (!session.isBlank()) {
                var finish = request(List.of()).put("write_session_id", session);
                var locations = start.path("locations");
                boolean dummy = true;
                for (var location : locations) {
                    String type = location.path("type").asText();
                    if (!type.equals("6") && !type.equals("ST_DUMMY")) {
                        dummy = false;
                        break;
                    }
                }
                finish.putObject("success_blocks").put("offset", dummy ? locations.size() : 0);
                post(target, "/api/finishWriteCache", finish, WRITE_TIMEOUT);
                if (!dummy) throw new IOException("Mock Remote requires a KVCM dummy storage candidate");
            } else if (!start.path("locations").isEmpty()) {
                throw new IOException("KVCM returned locations without a write session");
            }
            metrics.increment("mock_remote_cache_write_success");
        } catch (InterruptedException error) {
            Thread.currentThread().interrupt();
        } catch (IOException | RuntimeException error) {
            metrics.increment("mock_remote_cache_write_failure");
        }
    }

    private ObjectNode request(List<Long> keys) {
        var request = JSON.createObjectNode().put("trace_id", UUID.randomUUID().toString()).put("instance_id", instance);
        if (!keys.isEmpty()) {
            var blocks = request.putArray("block_keys");
            keys.forEach(key -> blocks.add(Long.toString(key)));
        }
        return request;
    }
}
