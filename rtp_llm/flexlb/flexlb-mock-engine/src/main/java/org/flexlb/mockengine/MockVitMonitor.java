package org.flexlb.mockengine;

import io.grpc.*;
import org.flexlb.enums.FlexMetricType;
import org.flexlb.metric.FlexMetricTags;
import org.flexlb.metric.FlexMonitor;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

/** Mock-specific RPC accounting; status polling is never image traffic. */
final class MockVitMonitor implements ServerInterceptor, AutoCloseable {
    private static final Set<String> IMAGE_METHODS = Set.of(
            "RemoteMultimodalEmbedding", "GetMultimodalHashes", "AsyncSubmitEmbedding");
    private final FlexMonitor sink;
    private final Map<String, String> labels;
    private final ScheduledExecutorService sampler;
    private final AtomicBoolean reportFailureLogged = new AtomicBoolean();
    private final Set<String> registered = new java.util.HashSet<>();
    private long previousComputations;
    private long previousNanos = System.nanoTime();

    MockVitMonitor(FlexMonitor sink, Map<String, String> labels) {
        this.sink = sink;
        this.labels = Map.copyOf(labels);
        for (String name : new String[]{"mock_vit_rpc_qps", "mock_vit_rpc_success_qps",
                "mock_vit_rpc_error_qps", "py_rtp_vit_qps", "py_rtp_vit_success_qps",
                "py_rtp_vit_error_qps", "py_rtp_vit_embedding_overload_qps",
                "py_rtp_vit_embedding_cache_hit_qps", "py_rtp_vit_embedding_cache_miss_qps",
                "py_rtp_vit_embedding_cache_inflight_qps", "py_rtp_vit_embedding_cache_eviction_qps"}) register(name, FlexMetricType.QPS);
        for (String name : new String[]{"mock_vit_active", "mock_vit_queued", "mock_vit_pending",
                "mock_vit_embedding_bytes", "mock_vit_hash_bytes", "mock_vit_computations_total",
                "mock_vit_computation_qps", "mock_vit_rpc_rt_us",
                "py_rtp_vit_embedding_cache_tokens", "py_rtp_vit_embedding_cache_bytes",
                "py_rtp_vit_embedding_queue_size", "py_rtp_vit_embedding_queue_wait_rt",
                "py_rtp_vit_preprocess_queue_size", "py_rtp_vit_preprocess_rt", "py_rtp_vit_download_rt", "py_rtp_vit_preprocess_other_rt",
                "py_rtp_vit_embedding_batch_size", "py_rtp_vit_embedding_forward_rt", "py_rtp_vit_embedding_rt"}) register(name, FlexMetricType.GAUGE);
        sampler = Executors.newSingleThreadScheduledExecutor(task -> {
            var thread = new Thread(task, "mock-vit-monitor");
            thread.setDaemon(true);
            return thread;
        });
    }

    void start(MockVitService service) {
        service.setSchedulerMetrics(values -> values.forEach((name, value) -> {
            var tags = new HashMap<>(labels);
            if (name.equals("py_rtp_vit_qps")) tags.put("source", "mm_embedding");
            report(name, tags, value.doubleValue());
        }));
        sampler.scheduleAtFixedRate(() -> {
            try { sample(service.snapshot(), System.nanoTime()); }
            catch (RuntimeException error) {
                System.err.println("Mock ViT monitoring sample failed: " + error.getClass().getSimpleName());
            }
        }, 0, 1, TimeUnit.SECONDS);
    }

    synchronized void sample(Map<String, Long> snapshot, long now) {
        double elapsed = Math.max(1e-9, (now - previousNanos) / 1e9);
        long computations = snapshot.get("computations");
        report("mock_vit_computation_qps", labels, Math.max(0, computations - previousComputations) / elapsed);
        previousComputations = computations;
        previousNanos = now;
        snapshot.forEach((name, value) -> {
            String metric = name.equals("computations") ? "mock_vit_computations_total" : "mock_vit_" + name;
            register(metric, FlexMetricType.GAUGE);
            report(metric, labels, value);
        });
    }

    private void register(String name, FlexMetricType type) {
        if (registered.add(name)) sink.register(name, type);
    }

    private void report(String name, Map<String, String> tags, double value) {
        // Observability failures must not change the RPC's result or stop the sampler.
        try { sink.report(name, new FlexMetricTags.ImmutableFlexMetricTags(tags), value); }
        catch (RuntimeException error) {
            if (reportFailureLogged.compareAndSet(false, true))
                System.err.println("Mock ViT metric report failed: " + name);
        }
    }

    @Override public <Q, S> ServerCall.Listener<Q> interceptCall(
            ServerCall<Q, S> call, Metadata headers, ServerCallHandler<Q, S> next) {
        String method = call.getMethodDescriptor().getBareMethodName();
        if (!IMAGE_METHODS.contains(method)) return next.startCall(call, headers);
        var tags = new HashMap<>(labels);
        tags.put("method", method);
        long started = System.nanoTime();
        var finished = new AtomicBoolean();
        report("mock_vit_rpc_qps", tags, 1);
        java.util.function.Consumer<Status> finish = status -> {
            if (!finished.compareAndSet(false, true)) return;
            var terminal = new HashMap<>(tags);
            terminal.put("code", status.getCode().name());
            report(status.isOk() ? "mock_vit_rpc_success_qps" : "mock_vit_rpc_error_qps", terminal, 1);
            report("mock_vit_rpc_rt_us", terminal, (System.nanoTime() - started) / 1000.0);
        };
        var wrapped = new ForwardingServerCall.SimpleForwardingServerCall<Q, S>(call) {
            @Override public void close(Status status, Metadata trailers) {
                finish.accept(status);
                super.close(status, trailers);
            }
        };
        try {
            return new ForwardingServerCallListener.SimpleForwardingServerCallListener<Q>(next.startCall(wrapped, headers)) {
                @Override public void onCancel() {
                    finish.accept(Status.CANCELLED);
                    super.onCancel();
                }
            };
        } catch (RuntimeException error) {
            finish.accept(Status.fromThrowable(error));
            throw error;
        }
    }

    @Override public void close() {
        sampler.shutdownNow();
        sink.close();
    }
}
