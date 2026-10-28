package org.flexlb.mock.client;

import java.util.*;
import org.flexlb.mock.api.Endpoint;
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
    record Start(String master, List<Input> requests, double qps, int concurrency, int queue_capacity,
                 long request_timeout_ms, long drain_timeout_ms) {
        public Start {
            requests = List.copyOf(requests);
            if (master == null || master.isBlank() || requests.isEmpty() || !Double.isFinite(qps) || qps <= 0
                    || requests.size() / qps > 86400 || concurrency < 1 || queue_capacity < 1 || queue_capacity > 100000
                    || request_timeout_ms < 1 || request_timeout_ms > 3600000 || drain_timeout_ms < 1 || drain_timeout_ms > 3600000)
                throw new IllegalArgumentException("Invalid finite playback plan (maximum 24 hours)");
        }
    }
    record Item(long request_id, long index, Input input) {
        public Item { if (request_id <= 0 || index < 0) throw new IllegalArgumentException("Invalid request identity"); }
    }
    record Shard(long sequence, String master, List<Item> requests, double qps, int concurrency,
                 int queue_capacity, long request_timeout_ms, long drain_timeout_ms) {
        public Shard {
            requests = List.copyOf(requests);
            if (sequence < 1 || requests.isEmpty()) throw new IllegalArgumentException("Empty/invalid shard");
            new Start(master, requests.stream().map(Item::input).toList(), qps, concurrency, queue_capacity, request_timeout_ms, drain_timeout_ms);
            if (requests.getLast().index() / qps > 86400) throw new IllegalArgumentException("Playback exceeds 24 hours");
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
