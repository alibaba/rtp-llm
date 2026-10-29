package org.flexlb.mock.client;

import java.util.*;
import org.flexlb.mock.api.Endpoint;
import org.flexlb.mock.api.Argument;
import com.fasterxml.jackson.annotation.JsonInclude;
import static org.flexlb.mock.api.Endpoint.Kind.*;

/** Private Client control transport. Python owns the Case API. */
public interface ClientProtocol {
    record Empty() {}
    record Input(List<Integer> token_ids, int max_new_tokens, int priority) {
        public Input {
            token_ids = List.copyOf(token_ids);
            if (token_ids.isEmpty() || token_ids.stream().anyMatch(v -> v < 0) || max_new_tokens < 1 || priority < 0)
                throw new IllegalArgumentException("Invalid text request");
        }
    }
    @JsonInclude(JsonInclude.Include.NON_NULL)
    record Start(@Argument(optional=true) List<String> masters, @Argument(optional=true) List<Input> requests,
                 @Argument(optional=true) String requests_file, double qps, int concurrency, int queue_capacity,
                 long request_timeout_ms, long drain_timeout_ms) {
        public Start {
            if ((requests == null) == (requests_file == null))
                throw new IllegalArgumentException("Specify exactly one of requests or requests_file");
            if (requests != null) {
                requests = List.copyOf(requests);
                if (requests.isEmpty() || requests.size() / qps > 86400)
                    throw new IllegalArgumentException("Invalid request count (maximum 24 hours)");
            }
            if (requests_file != null && (requests_file.isBlank() || !java.nio.file.Path.of(requests_file).isAbsolute()))
                throw new IllegalArgumentException("requests_file must be an absolute local path");
            if (masters != null) {
                masters = List.copyOf(masters);
                if (masters.isEmpty() || new HashSet<>(masters).size() != masters.size() || masters.stream().anyMatch(String::isBlank))
                    throw new IllegalArgumentException("masters must be nonempty and unique when supplied");
            }
            if (!Double.isFinite(qps) || qps <= 0 || concurrency < 1 || queue_capacity < 1 || queue_capacity > 100000
                    || request_timeout_ms < 1 || request_timeout_ms > 3600000 || drain_timeout_ms < 1 || drain_timeout_ms > 3600000)
                throw new IllegalArgumentException("Invalid finite playback plan (maximum 24 hours)");
        }
    }
    record Item(long request_id, long index, Input input) {
        public Item { if (request_id <= 0 || index < 0) throw new IllegalArgumentException("Invalid request identity"); }
    }
    @JsonInclude(JsonInclude.Include.NON_NULL)
    record Shard(long sequence, @Argument(optional=true) List<String> masters, String requests_file,
                 long count, long first_index, int stride, long first_request_id, String sha256,
                 double qps, int concurrency, int queue_capacity, long request_timeout_ms, long drain_timeout_ms) {
        public Shard {
            new Start(masters, null, requests_file, qps, concurrency, queue_capacity, request_timeout_ms, drain_timeout_ms);
            masters = masters == null ? null : List.copyOf(masters);
            if (sequence < 1 || count < 1 || first_index < 0 || stride < 1 || first_index >= stride || first_request_id < 1
                    || sha256 == null || !sha256.matches("[0-9a-f]{64}"))
                throw new IllegalArgumentException("Invalid shard");
            long last = Math.addExact(first_index, Math.multiplyExact(count - 1, stride));
            Math.addExact(first_request_id, last);
            if (last / qps > 86400) throw new IllegalArgumentException("Playback exceeds 24 hours");
        }
    }
    record Begin(long sequence, long start_epoch_ms) {}
    record Sequence(long sequence) {}
    record Ack(boolean accepted) {}
    record State(boolean done, boolean stopped, long produced, long issued, long finished, long unsent,
                 int queued, int inflight, Map<String, Long> outcomes, List<String> errors) {}
    record Summary(Map<String, State> clients, List<String> errors) {}

    @Endpoint(value="client/start", kind=action) Ack start(Start args) throws Exception;
    @Endpoint(value="client/stop", kind=action) Summary stop(Empty args) throws Exception;
    @Endpoint(value="worker/prepare", kind=action) Ack prepare(Shard args) throws Exception;
    @Endpoint(value="worker/begin", kind=action) Ack begin(Begin args);
    @Endpoint(value="worker/halt", kind=action) Ack halt(Sequence args);
    @Endpoint(value="worker/state", kind=read) State state(Sequence args);
}
