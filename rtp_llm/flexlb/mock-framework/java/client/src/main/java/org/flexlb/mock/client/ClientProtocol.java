package org.flexlb.mock.client;

import java.util.*;
import org.flexlb.mock.protocol.ControlServer.OptionalArgument;
import com.fasterxml.jackson.annotation.JsonInclude;

/**
 * Defines the control messages exchanged by the Case runner and replay Clients.
 */
public final class ClientProtocol {
    private ClientProtocol() {}

    private static List<String> validateReplay(List<String> masters, String requestsFile, Double qps, Double replaySpeed,
                                               int concurrency, int queueCapacity, long requestTimeoutMs,
                                               long drainTimeoutMs, Integer fallbackMaxOutputTokens) {
        if (requestsFile == null || requestsFile.isBlank()
                || !java.nio.file.Path.of(requestsFile).isAbsolute() || !requestsFile.endsWith(".mckidx"))
            throw new IllegalArgumentException("requests_file must be an absolute .mckidx path");
        if (masters != null) {
            masters = List.copyOf(masters);
            if (masters.isEmpty() || new HashSet<>(masters).size() != masters.size() || masters.stream().anyMatch(String::isBlank))
                throw new IllegalArgumentException("masters must be nonempty and unique when supplied");
        }
        if (fallbackMaxOutputTokens != null && fallbackMaxOutputTokens < 1)
            throw new IllegalArgumentException("fallback_max_output_tokens must be positive");
        Double rate = qps != null ? qps : replaySpeed;
        if ((qps == null) == (replaySpeed == null) || !Double.isFinite(rate) || rate <= 0
                || concurrency < 1 || queueCapacity < 1 || queueCapacity > 100000
                || requestTimeoutMs < 1 || requestTimeoutMs > 3600000 || drainTimeoutMs < 1 || drainTimeoutMs > 3600000)
            throw new IllegalArgumentException("Invalid finite playback plan (maximum 24 hours)");
        return masters;
    }

    public record Empty() {}

    /**
     * Starts one replay using a concurrency budget shared by all Clients.
     */
    @JsonInclude(JsonInclude.Include.NON_NULL)
    public record Start(@OptionalArgument List<String> masters, String requests_file,
                 @OptionalArgument Double qps, @OptionalArgument Double replay_speed,
                 int total_concurrency, int queue_capacity,
                 long request_timeout_ms, long drain_timeout_ms,
                 @OptionalArgument Boolean reuse_cache,
                 @OptionalArgument Integer fallback_max_output_tokens,
                 @OptionalArgument Warmup warmup) {
        public Start(List<String> masters, String requests_file, Double qps, Double replay_speed,
                     int total_concurrency, int queue_capacity, long request_timeout_ms, long drain_timeout_ms) {
            this(masters, requests_file, qps, replay_speed, total_concurrency, queue_capacity,
                request_timeout_ms, drain_timeout_ms, false, null, null);
        }
        public Start {
            masters = validateReplay(masters, requests_file, qps, replay_speed, total_concurrency, queue_capacity,
                request_timeout_ms, drain_timeout_ms, fallback_max_output_tokens);
        }
    }

    /**
     * Assigns one Client its request ordinals and concurrency budget for a replay.
     */
    @JsonInclude(JsonInclude.Include.NON_NULL)
    public record ClientReplayPlan(long sequence, @OptionalArgument List<String> masters, String requests_file,
                 long count, long first_index, int stride, long first_request_id, String sha256,
                 @OptionalArgument Double qps, @OptionalArgument Double replay_speed,
                 int concurrency, int queue_capacity, long request_timeout_ms, long drain_timeout_ms,
                 @OptionalArgument Boolean reuse_cache,
                 @OptionalArgument Integer fallback_max_output_tokens,
                 @OptionalArgument Warmup warmup) {
        public ClientReplayPlan(long sequence, List<String> masters, String requests_file, long count, long first_index,
                     int stride, long first_request_id, String sha256, Double qps, Double replay_speed,
                     int concurrency, int queue_capacity, long request_timeout_ms, long drain_timeout_ms) {
            this(sequence, masters, requests_file, count, first_index, stride, first_request_id, sha256, qps,
                replay_speed, concurrency, queue_capacity, request_timeout_ms, drain_timeout_ms, false, null, null);
        }
        public ClientReplayPlan {
            masters = validateReplay(masters, requests_file, qps, replay_speed, concurrency, queue_capacity,
                request_timeout_ms, drain_timeout_ms, fallback_max_output_tokens);
            if (sequence < 1 || count < 1 || first_index < 0 || stride < 1 || first_index >= stride || first_request_id < 1
                    || sha256 == null || !sha256.matches("[0-9a-f]{64}"))
                throw new IllegalArgumentException("Invalid worker playback plan");
            long last = Math.addExact(first_index, Math.multiplyExact(count - 1, stride));
            Math.addExact(first_request_id, last);
        }
    }

    /**
     * Scales replay progress for a fixed amount of elapsed time from the common start.
     */
    public record Warmup(long duration_ms, double rate_scale) {
        public Warmup {
            if (duration_ms < 1 || duration_ms > 86_400_000 || !Double.isFinite(rate_scale)
                    || rate_scale <= 0 || rate_scale >= 1)
                throw new IllegalArgumentException("Warmup requires duration_ms in [1,86400000] and 0 < rate_scale < 1");
        }
    }

    public record Begin(long sequence, long start_epoch_ms) {}

    public record Sequence(long sequence) {}

    public record Ack(boolean accepted) {}

    public record State(boolean done, boolean stopped, long produced, long issued, long finished, long unissued,
                 int queued, int inflight, Map<String, Long> outcomes, List<String> errors) {}

    public record Summary(Map<String, State> clients, List<String> errors) {}
}
