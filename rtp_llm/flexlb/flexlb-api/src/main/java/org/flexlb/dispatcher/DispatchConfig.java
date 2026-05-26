package org.flexlb.dispatcher;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import lombok.Getter;
import lombok.Setter;
import org.flexlb.config.EnvConfigOverrides;
import org.flexlb.util.JsonUtils;

import java.util.Map;

/**
 * Operator-facing tuning surface for the dispatcher. Six fields and nothing else — every
 * timeout/safety knob that "no one actually tunes" lives as a constant inside
 * {@code DispatcherConfiguration} / {@code WebClientPassthroughClient} / {@code WebClientFeClient}
 * (see those classes for FE_CONNECT_TIMEOUT_MS / FE_PENDING_ACQUIRE_TIMEOUT_MS /
 * STREAM_TIMEOUT_MS / MAX_RESPONSE_BYTES).
 *
 * <p>Loading order: defaults → JSON from {@code DISPATCH_CONFIG} env → per-field env overrides
 * (e.g. {@code DISPATCH_BATCH_TIMEOUT_MS}). The per-field env wins, matching the {@code FLEXLB_CONFIG}
 * contract operators already know.
 *
 * <p>Unknown JSON properties are ignored so a stale {@code DISPATCH_CONFIG} carrying old field
 * names (subBatchSize, feRequestTimeoutMs, …) still boots — they just have no effect.
 */
@Getter
@Setter
@JsonIgnoreProperties(ignoreUnknown = true)
public class DispatchConfig {

    /** Master kill switch. {@code false} → {@code /dispatcher/**} routes are not registered. */
    private boolean enabled = false;

    /**
     * Chunk splitting DSL. {@code count:N} → exactly N chunks (default). {@code size:N} →
     * each chunk holds at most N items. Bare integer is shorthand for {@code size:N}.
     * Parsed eagerly during {@link #validate()} so a malformed value fails fast at boot.
     */
    private String subBatch = "count:5";

    /** Service-discovery name for the FE pool. Required when {@link #enabled} is true. */
    private String fePoolServiceId = "";

    /**
     * Per batch sub-call: how long to wait for the FE to start responding (first byte). Stops once
     * the response header arrives — body read is bounded by infra defaults, not this. Same idea as
     * ft_proxy's {@code -t}, but only covers the header-wait window in reactor-netty's model.
     */
    private int batchTimeoutMs = 5000;

    /**
     * Max concurrent TCP connections <strong>per FE host</strong> (not total across the pool).
     * Reactor-netty's {@code ConnectionProvider} pools per remote address, so with N FE hosts the
     * effective ceiling is {@code feMaxConnectionsPerHost × N}. Tune from target QPS × avg request
     * time / FE count, with safety margin.
     */
    private int feMaxConnectionsPerHost = 200;

    /**
     * Max pending acquires <strong>per FE host</strong> when the connection pool is exhausted.
     * Acts as a backpressure ring buffer; exceeding it makes the dispatcher fail fast instead of
     * piling up an unbounded queue under overload.
     */
    private int feMaxPendingAcquirePerHost = 1000;

    /** Internal cache of the parsed sub-batch spec — populated in {@link #validate()}. */
    private transient SubBatchSpec subBatchSpec;

    public static DispatchConfig fromJson(String json) {
        return fromJsonWithEnv(json, System.getenv());
    }

    /**
     * Test seam: load with an explicit env map instead of {@link System#getenv()}.
     */
    public static DispatchConfig fromJsonWithEnv(String json, Map<String, String> env) {
        DispatchConfig c = (json == null || json.isBlank())
                ? new DispatchConfig()
                : JsonUtils.toObject(json, DispatchConfig.class);
        EnvConfigOverrides.apply(c, "DISPATCH_", env);
        c.validate();
        return c;
    }

    public SubBatchSpec subBatchSpec() {
        return subBatchSpec;
    }

    private void validate() {
        if (enabled && (fePoolServiceId == null || fePoolServiceId.isBlank())) {
            throw new IllegalArgumentException("DISPATCH_CONFIG.enabled=true requires fePoolServiceId");
        }
        if (batchTimeoutMs <= 0) {
            throw new IllegalArgumentException("batchTimeoutMs must be > 0, got " + batchTimeoutMs);
        }
        if (feMaxConnectionsPerHost <= 0) {
            throw new IllegalArgumentException(
                    "feMaxConnectionsPerHost must be > 0, got " + feMaxConnectionsPerHost);
        }
        if (feMaxPendingAcquirePerHost <= 0) {
            throw new IllegalArgumentException(
                    "feMaxPendingAcquirePerHost must be > 0, got " + feMaxPendingAcquirePerHost);
        }
        // SubBatchSpec.parse throws IllegalArgumentException with a precise message on bad DSL.
        this.subBatchSpec = SubBatchSpec.parse(subBatch);
    }
}
