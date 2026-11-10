package org.flexlb.mock.client;

import java.util.*;
import org.flexlb.mock.protocol.Argument;
import com.fasterxml.jackson.annotation.JsonInclude;

/**
 * Defines the control messages exchanged by the Case runner and replay Clients.
 */
public interface ClientProtocol {
    static long requestCount(long total, long firstIndex, int stride) {
        return total / stride + (firstIndex < total % stride ? 1 : 0);
    }

    Ack start(Start args) throws Exception;

    Summary stop(Empty args) throws Exception;

    Ack prepare(ClientReplayPlan args) throws Exception;

    Ack begin(Begin args);

    Ack halt(Sequence args);

    State state(Sequence args);

    record Empty() {}

    @JsonInclude(JsonInclude.Include.NON_NULL)
    record Start(@Argument(optional=true) List<String> masters, String requests_file,
                 @Argument(optional=true) Double qps, @Argument(optional=true) Double replay_speed,
                 int concurrency, int queue_capacity,
                 long request_timeout_ms, long drain_timeout_ms,
                 @Argument(optional=true) Boolean reuse_cache,
                 @Argument(optional=true) Integer fallback_max_output_tokens) {
        public Start(List<String> masters, String requests_file, Double qps, Double replay_speed,
                     int concurrency, int queue_capacity, long request_timeout_ms, long drain_timeout_ms) {
            this(masters, requests_file, qps, replay_speed, concurrency, queue_capacity,
                request_timeout_ms, drain_timeout_ms, false, null);
        }
        public Start {
            if (requests_file == null || requests_file.isBlank() || !java.nio.file.Path.of(requests_file).isAbsolute() || !requests_file.endsWith(".mckidx"))
                throw new IllegalArgumentException("requests_file must be an absolute .mckidx path");
            if (masters != null) {
                masters = List.copyOf(masters);
                if (masters.isEmpty() || new HashSet<>(masters).size() != masters.size() || masters.stream().anyMatch(String::isBlank))
                    throw new IllegalArgumentException("masters must be nonempty and unique when supplied");
            }
            if (fallback_max_output_tokens != null && fallback_max_output_tokens < 1)
                throw new IllegalArgumentException("fallback_max_output_tokens must be positive");
            if ((qps == null) == (replay_speed == null)
                    || !Double.isFinite(qps != null ? qps : replay_speed) || (qps != null ? qps : replay_speed) <= 0 || concurrency < 1 || queue_capacity < 1 || queue_capacity > 100000
                    || request_timeout_ms < 1 || request_timeout_ms > 3600000 || drain_timeout_ms < 1 || drain_timeout_ms > 3600000)
                throw new IllegalArgumentException("Invalid finite playback plan (maximum 24 hours)");
        }
    }

    /**
     * Assigns one Client its request ordinals and concurrency budget for a replay.
     */
    @JsonInclude(JsonInclude.Include.NON_NULL)
    record ClientReplayPlan(long sequence, @Argument(optional=true) List<String> masters, String requests_file,
                 long count, long first_index, int stride, long first_request_id, String sha256,
                 @Argument(optional=true) Double qps, @Argument(optional=true) Double replay_speed,
                 int concurrency, int queue_capacity, long request_timeout_ms, long drain_timeout_ms,
                 @Argument(optional=true) Boolean reuse_cache,
                 @Argument(optional=true) Integer fallback_max_output_tokens) {
        public ClientReplayPlan(long sequence, List<String> masters, String requests_file, long count, long first_index,
                     int stride, long first_request_id, String sha256, Double qps, Double replay_speed,
                     int concurrency, int queue_capacity, long request_timeout_ms, long drain_timeout_ms) {
            this(sequence, masters, requests_file, count, first_index, stride, first_request_id, sha256, qps,
                replay_speed, concurrency, queue_capacity, request_timeout_ms, drain_timeout_ms, false, null);
        }
        public ClientReplayPlan {
            masters = new Start(masters, requests_file, qps, replay_speed, concurrency, queue_capacity,
                request_timeout_ms, drain_timeout_ms, reuse_cache, fallback_max_output_tokens).masters();
            if (sequence < 1 || count < 1 || first_index < 0 || stride < 1 || first_index >= stride || first_request_id < 1
                    || sha256 == null || !sha256.matches("[0-9a-f]{64}"))
                throw new IllegalArgumentException("Invalid worker playback plan");
            long last = Math.addExact(first_index, Math.multiplyExact(count - 1, stride));
            Math.addExact(first_request_id, last);
        }
    }

    record Begin(long sequence, long start_epoch_ms) {}

    record Sequence(long sequence) {}

    record Ack(boolean accepted) {}

    record State(boolean done, boolean stopped, long produced, long issued, long finished, long unsent,
                 int queued, int inflight, Map<String, Long> outcomes, List<String> errors) {}

    record Summary(Map<String, State> clients, List<String> errors) {}
}
