package org.flexlb.mockengine;

import com.google.protobuf.ByteString;
import io.grpc.Context;
import io.grpc.Status;
import io.grpc.stub.StreamObserver;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.engine.grpc.MultimodalRpcServiceGrpc;

import java.io.ByteArrayOutputStream;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.UUID;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;

/** Main ViT wire contract with deterministic forward, byte-bounded caches and shared computation. */
public final class MockVitService extends MultimodalRpcServiceGrpc.MultimodalRpcServiceImplBase implements AutoCloseable {
    public record Config(int hiddenSize, int concurrency, int queueCapacity, long embeddingCacheBytes,
                         long hashCacheBytes, int maxInputBytes, long maxOutputBytes, double computeDelayMs) {
        public Config {
            if (hiddenSize <= 0 || concurrency <= 0 || queueCapacity < 0 || embeddingCacheBytes < 0
                    || hashCacheBytes < 0 || maxInputBytes <= 0 || maxOutputBytes <= 0 || !Double.isFinite(computeDelayMs) || computeDelayMs < 0)
                throw new IllegalArgumentException("invalid Mock ViT limits");
        }
        public static Config defaults() {
            // Resolved server defaults from VitConfig; width, output guard and delay are Mock calibration.
            return new Config(64, 64, 64, 2L << 30, 256L << 20, 100 << 20, 64L << 20, 1);
        }
    }
    final Config config;
    private final MockV41Image.Config v41;
    final String instance = UUID.randomUUID().toString().replace("-", "");
    private final ThreadPoolExecutor executor;
    java.util.function.Consumer<Runnable> submitAsync;
    private final Map<String, CompletableFuture<MockVitImage.Result>> pending = new ConcurrentHashMap<>();
    private final MockVitEmbeddingCache<MockVitImage.Result> embeddingCache;
    private final MockVitHashCache hashCache;
    private final long embeddingTimeoutMs;
    private final int maxImageBytes;

    private final MockVitScheduler gpuScheduler;
    private final MockVitPreprocessor preprocessor;
    private final Map<String, String> downloadHeaders;
    private final com.fasterxml.jackson.databind.JsonNode effectiveConfig;
    private static final class Task {
        final String key;
        final String generation = UUID.randomUUID().toString().replace("-", "");
        final CompletableFuture<MockVitImage.Result> result = new CompletableFuture<>();
        final java.util.Set<Long> requests = new java.util.HashSet<>();
        java.util.concurrent.FutureTask<Void> worker;
        boolean errorReported;
        Task(String key) { this.key = key; }
    }
    private record Claims(List<CompletableFuture<MockVitImage.Result>> values, java.util.Collection<Task> tasks) {}
    private final Map<CompletableFuture<MockVitImage.Result>, Task> tasks = new ConcurrentHashMap<>();
    private final Map<Long, java.util.Set<Task>> owners = new java.util.HashMap<>();
    private volatile java.util.function.Consumer<Map<String, Number>> metrics = ignored -> {};
    private final boolean cacheEnabled;
    private volatile boolean closed;
    java.util.function.BooleanSupplier alive = () -> !closed;
    final AtomicLong computations = new AtomicLong();

    MockVitService(Config config, MockV41Image.Config model, com.fasterxml.jackson.databind.JsonNode document) {
        this.config = config;
        this.v41 = model;
        typedCacheCapacity = model == null ? 0 : Math.max(0, Integer.parseInt(
                System.getenv().getOrDefault("V41_IMAGE_ENCODE_CACHE_ITEMS", "16")));
        executor = new ThreadPoolExecutor(config.concurrency(), config.concurrency(), 0, TimeUnit.MILLISECONDS,
                new java.util.concurrent.LinkedBlockingQueue<>(), task -> {
                    Thread thread = new Thread(task, "mock-vit-forward"); thread.setDaemon(true); return thread;
                }, new ThreadPoolExecutor.AbortPolicy());
        submitAsync = executor::execute;
        effectiveConfig = document.deepCopy();
        var limits = document.path("roles").path("vit");
        if (!limits.path("greennet_enabled").isBoolean() || limits.path("greennet_enabled").booleanValue())
            throw new IllegalArgumentException("GreenNet enablement must be explicit and inspection is unsupported");
        if (config.concurrency() != limits.path("vit_concurrency").intValue()
                || config.queueCapacity() != limits.path("vit_max_queue_size").intValue()
                || config.embeddingCacheBytes() != limits.path("mm_cache_cpu_max_bytes").longValue()
                || config.hashCacheBytes() != limits.path("mm_hash_key_cache_max_bytes").longValue())
            throw new IllegalArgumentException("ViT runtime limits differ from effective config");
        if (MockEngineConfig.vitWorkBudget(limits) != null)
            throw new IllegalArgumentException("configured ViT model has no work estimator; cannot execute a nonnull work_budget");
        cacheEnabled = limits.path("mm_cache_gpu_max_bytes").longValue() > 0 || config.embeddingCacheBytes() > 0;
        embeddingTimeoutMs = limits.path("mm_timeout_ms").longValue();
        maxImageBytes = MockEngineConfig.vitImageBytes(limits);
        preprocessor = new MockVitPreprocessor(limits.path("use_local_preprocess").booleanValue(),
                limits.path("mm_preprocess_max_workers").intValue());
        downloadHeaders = MockEngineConfig.vitDownloadHeaders(limits);
        embeddingCache = new MockVitEmbeddingCache<>(limits.path("mm_cache_gpu_max_bytes").longValue(), config.embeddingCacheBytes());
        embeddingCache.tokens = MockVitImage.Result::rows;
        hashCache = new MockVitHashCache(config.hashCacheBytes(), limits.path("hash_cache_accounting"));
        preprocessor.configureMediaCache(limits.path("url_cache_item_num").intValue());
        gpuScheduler = new MockVitScheduler(limits.path("mm_max_queue_size").intValue(),
                limits.path("gpu_max_batch_size").intValue(), limits.path("gpu_max_batch_images").longValue(),
                limits.path("gpu_batch_wait_ms").longValue(), config.computeDelayMs(), MockEngineConfig.vitWorkBudget(limits));
    }

    void timing(MockTiming timing) { gpuScheduler.timing(timing); }
    void forwardMs(double value) { gpuScheduler.forwardMs(value); }

    void reportTiming(Map<String, Number> values) {
        try { metrics.accept(values); } catch (RuntimeException ignored) { /* telemetry owns no work */ }
    }
    private void report(String name, Number value) {
        try { metrics.accept(Map.of(name, value)); } catch (RuntimeException ignored) { /* telemetry owns no work */ }
    }
    void setSchedulerMetrics(java.util.function.Consumer<Map<String, Number>> reporter) {
        metrics = reporter;
        embeddingCache.setMetrics(reporter);
        gpuScheduler.setMetrics(reporter);
        preprocessor.setMetrics(reporter);
    }

    private CompletableFuture<MockVitImage.Result> cached(String key, long budgetMs) {
        var read = embeddingCache.claim(key, budgetMs);
        if (read == null) return null;
        report("py_rtp_vit_embedding_cache_hit_qps", 1);
        var result = new CompletableFuture<MockVitImage.Result>();
        // Cache consumption belongs to the caller, outside shared admission/queue locks.
        Thread.startVirtualThread(() -> {
            try {
                var hit = read.get();
                consumeHashes(key, hit.value(), hit.generation());
                result.complete(hit.value());
            } catch (Throwable error) { result.completeExceptionally(error); }
        });
        return result;
    }
    private MockVitHashCache.Entry hash(String key) { return hashCache.peek(key); }
    private void consumeHashes(String key, MockVitImage.Result value, String generation) {
        if (!closed && hashCache.consume(key, generation) == null) hashCache.put(key, value, generation);
    }
    private void remember(String key, MockVitImage.Result result, String generation) {
        if (closed || key.startsWith("uncached:")) return;
        // Feature hashes are an independent CPU sidecar, not embedding-pool payload.
        var layout = List.of(new MockVitEmbeddingCache.Tensor(result.embedding(), result.embeddingBytes(),
                result.embedding().getDataType() == EngineRpcService.TensorPB.DataType.BF16 ? 2 : 4,
                MockVitEmbeddingCache.Device.GPU));
        embeddingCache.put(key, result, layout, generation);
        if (embeddingCache.contains(key)) hashCache.put(key, result, generation);
    }

    CompletableFuture<MockVitImage.Result> acquire(EngineRpcService.MultimodalInputPB image, long timeoutMs) {
        try { return claim(List.of(image), 0, timeoutMs, true).values().getFirst(); }
        catch (Throwable error) { return CompletableFuture.failedFuture(error); }
    }

    // Claims, owner registration, batch admission and submission share the cancellation lock.
    private synchronized Claims claim(
            List<EngineRpcService.MultimodalInputPB> images, long requestId, Long queueTimeout, boolean consume) {
        if (closed) throw Status.UNAVAILABLE.withDescription("MMProcessEngine stopped").asRuntimeException();
        if (Context.current().isCancelled()) throw MockVitErrors.cancelled();
        var results = new ArrayList<CompletableFuture<MockVitImage.Result>>();
        var claims = new java.util.LinkedHashSet<Task>();
        var fresh = new LinkedHashMap<Task, EngineRpcService.MultimodalInputPB>();
        int admitted = tasks.size();
        var submitted = new java.util.HashSet<Task>();
        try {
            for (var image : images) {
                String key = MockVitImage.key(image);
                if (cacheEnabled) {
                    CompletableFuture<MockVitImage.Result> hit;
                    if (consume) hit = cached(key, queueTimeout == null ? embeddingTimeoutMs : queueTimeout);
                    else {
                        String generation = embeddingCache.touch(key);
                        if (generation != null) hashCache.consume(key, generation);
                        hit = generation == null ? null : CompletableFuture.completedFuture(null);
                    }
                    if (hit != null) {
                        if (!consume) report("py_rtp_vit_embedding_cache_hit_qps", 1);
                        results.add(hit); continue;
                    }
                }
                var shared = cacheEnabled ? pending.get(key) : null;
                Task task = shared == null ? null : tasks.get(shared);
                report(task == null ? "py_rtp_vit_embedding_cache_miss_qps" : "py_rtp_vit_embedding_cache_inflight_qps", 1);
                if (task == null) {
                    task = new Task(key); tasks.put(task.result, task); fresh.put(task, image);
                    if (cacheEnabled) pending.put(key, task.result);
                }
                task.requests.add(requestId); claims.add(task);
                owners.computeIfAbsent(requestId, ignored -> new java.util.LinkedHashSet<>()).add(task);
                Task owned = task;
                results.add(consume ? task.result.thenApply(value -> {
                    consumeHashes(key, value, owned.generation); return value;
                }) : task.result);
            }
            if (admitted + fresh.size() > (long)config.concurrency() + config.queueCapacity())
                throw MockRpcErrors.error(Status.RESOURCE_EXHAUSTED, 409, "[CONCURRENCY_LIMIT_ERROR] ViT async batch exceeds admission capacity");
            long lifetime = queueTimeout == null ? fresh.values().stream()
                    .mapToLong(image -> image.getMmPreprocessConfig().getMmTimeoutMs()).filter(value -> value > 0)
                    .max().orElse(embeddingTimeoutMs) : queueTimeout;
            long deadline = System.nanoTime() + TimeUnit.MILLISECONDS.toNanos(Math.max(0, lifetime));
            for (var item : fresh.entrySet()) {
                Task task = item.getKey(); var image = item.getValue();
                task.worker = new java.util.concurrent.FutureTask<Void>(() -> {
                    compute(task, image, deadline); return null;
                }) {
                    @Override protected void done() {
                        if (isCancelled() && !task.result.isDone()) task.result.completeExceptionally(failTask(task, MockVitErrors.cancelled()));
                        forget(task);
                    }
                };
                submitAsync.accept(task.worker); submitted.add(task);
            }
            return new Claims(results, List.copyOf(claims));
        } catch (Throwable error) {
            var failure = failed(error);
            for (var task : fresh.keySet()) if (!submitted.contains(task)) {
                task.result.completeExceptionally(failure); forget(task);
            }
            cancelRequest(requestId, claims);
            throw failure;
        }
    }

    private void compute(Task task, EngineRpcService.MultimodalInputPB image, long deadline) {
        try {
            report("py_rtp_vit_qps", 1);
            long media = image.getMmPreprocessConfig().getMmTimeoutMs() > 0
                    ? image.getMmPreprocessConfig().getMmTimeoutMs() : embeddingTimeoutMs;
            java.util.function.LongSupplier budget = () -> Math.min(media, remaining(deadline));
            var prepared = preprocessor.await(preprocessor.submit(() -> MockVitImage.prepare(image, config.hiddenSize(),
                    maxImageBytes, config.maxOutputBytes(), positiveBudget(budget), downloadHeaders, preprocessor.mediaCache())),
                    positiveBudget(budget));
            var gpuWork = gpuScheduler.submit(() -> MockVitImage.forward(prepared));
            MockVitImage.Result value;
            try { value = gpuWork.get(positiveBudget(budget), TimeUnit.MILLISECONDS); }
            catch (java.util.concurrent.TimeoutException | InterruptedException error) { gpuScheduler.cancelRequest(gpuWork); throw error; }
            report("py_rtp_vit_success_qps", 1);
            computations.incrementAndGet(); remember(task.key, value, task.generation); task.result.complete(value);
        } catch (Throwable error) { task.result.completeExceptionally(failTask(task, error)); }
    }
    private static long remaining(long deadline) { return TimeUnit.NANOSECONDS.toMillis(deadline - System.nanoTime()); }
    private static long positiveBudget(java.util.function.LongSupplier budget) throws java.util.concurrent.TimeoutException {
        long value = budget.getAsLong();
        if (value <= 0) throw new java.util.concurrent.TimeoutException("ViT request deadline exceeded");
        return value;
    }
    private synchronized void forget(Task task) {
        tasks.remove(task.result, task); pending.remove(task.key, task.result);
        for (long request : task.requests) {
            var set = owners.get(request);
            if (set != null) { set.remove(task); if (set.isEmpty()) owners.remove(request); }
        }
    }
    synchronized void cancelRequest(long requestId) { cancelRequest(requestId, List.copyOf(owners.getOrDefault(requestId, java.util.Set.of()))); }
    private synchronized void cancelRequest(long requestId, java.util.Collection<Task> claims) {
        var owned = owners.get(requestId);
        if (owned == null) return;
        for (var task : claims) {
            if (!owned.remove(task)) continue;
            task.requests.remove(requestId);
            if (task.requests.isEmpty() && (task.worker == null || executor.remove(task.worker))) {
                task.result.completeExceptionally(failTask(task, MockVitErrors.cancelled()));
                if (task.worker != null) task.worker.cancel(false);
                forget(task);
            }
        }
        if (owned.isEmpty()) owners.remove(requestId);
    }
    private synchronized io.grpc.StatusRuntimeException failTask(Task task, Throwable error) {
        var failure = MockVitErrors.normalize(cause(error));
        if (!task.errorReported && !MockVitErrors.reported(failure)) report("py_rtp_vit_error_qps", 1);
        task.errorReported = true;
        return MockVitErrors.markReported(failure);
    }
    private synchronized io.grpc.StatusRuntimeException requestFailure(long requestId, Throwable error) {
        if (error instanceof MetadataOverflow) return Status.RESOURCE_EXHAUSTED.withDescription(error.getMessage()).asRuntimeException();
        if (MockVitErrors.reported(error)) return MockVitErrors.normalize(error);
        var owned = owners.get(requestId);
        return owned == null || owned.isEmpty() ? failed(error) : failTask(owned.iterator().next(), error);
    }
    private io.grpc.StatusRuntimeException failed(Throwable error) {
        var failure = MockVitErrors.normalize(cause(error));
        if (!MockVitErrors.reported(failure)) {
            report("py_rtp_vit_error_qps", 1);
            failure = MockVitErrors.markReported(failure);
        }
        return failure;
    }

    static long timeout(Context context, long requested) {
        long remaining = context.getDeadline() == null ? 120000 : context.getDeadline().timeRemaining(TimeUnit.MILLISECONDS);
        return Math.min(requested > 0 ? requested : 120000, remaining);
    }
    private static Throwable cause(Throwable error) {
        while ((error instanceof CompletionException || error instanceof java.util.concurrent.ExecutionException) && error.getCause() != null) error = error.getCause();
        return error;
    }
    private <T> void respond(CompletableFuture<T> work, StreamObserver<T> observer, long budget, long requestId) {
        Context context = Context.current();
        var waiting = new CompletableFuture<T>();
        Context.CancellationListener cancelled = ignored -> waiting.completeExceptionally(
                Status.CANCELLED.withDescription("ViT caller cancelled").asRuntimeException());
        context.addListener(cancelled, Runnable::run);
        if (budget <= 0) waiting.completeExceptionally(Status.DEADLINE_EXCEEDED.asRuntimeException());
        else waiting.orTimeout(budget, TimeUnit.MILLISECONDS);
        work.whenComplete((value, error) -> {
            if (error == null) waiting.complete(value); else waiting.completeExceptionally(error);
        });
        waiting.whenComplete((value, error) -> {
            context.removeListener(cancelled);
            var failure = error == null ? null : requestFailure(requestId, cause(error));
            if (error != null && !work.isDone()) work.cancel(false);
            if (context.isCancelled()) return;
            if (error == null) { observer.onNext(value); observer.onCompleted(); }
            else {
                observer.onError(failure);
            }
        });
    }

    private static void validate(EngineRpcService.MultimodalInputsPB inputs) {
        if (inputs.hasV41Inputs()) throw new IllegalArgumentException("packed V4.1 images require a model-specific Mock ViT backend");
        if (inputs.getMultimodalInputsCount() == 0 || inputs.getMultimodalInputsCount() > 256)
            throw new IllegalArgumentException("expected 1..256 image inputs");
        for (var input : inputs.getMultimodalInputsList())
            if (input.getMultimodalUrl().isBlank() || input.getMultimodalType() != 1)
                throw new IllegalArgumentException("Mock ViT accepts URL IMAGE inputs only");
    }
    private CompletableFuture<List<MockVitImage.Result>> acquireAll(EngineRpcService.MultimodalInputsPB inputs, long budget) {
        validate(inputs);
        var claims = claim(inputs.getMultimodalInputsList(), inputs.getRequestId(), budget, true);
        var result = CompletableFuture.allOf(claims.values().toArray(CompletableFuture[]::new))
                .thenApply(ignored -> claims.values().stream().map(CompletableFuture::join).toList());
        result.whenComplete((value, error) -> { if (error != null) cancelRequest(inputs.getRequestId(), claims.tasks()); });
        return result;
    }

    // The typed model owns an item-count LRU inside forward, separate from the URL caches.
    private final Map<String, MockVitImage.Result> typedEmbeddings = new LinkedHashMap<>(16, .75f, true);
    private final int typedCacheCapacity;

    private synchronized MockVitImage.Result forwardTyped(MockV41Image.Work item) throws Exception {
        boolean trusted = !item.key().startsWith("uncached:");
        var resident = trusted ? typedEmbeddings.get(item.key()) : null;
        if (resident != null) return resident;
        var result = MockV41Image.compute(item.image(), v41, config.maxOutputBytes());
        computations.incrementAndGet();
        if (trusted && typedCacheCapacity > 0) {
            typedEmbeddings.put(item.key(), result);
            while (typedEmbeddings.size() > typedCacheCapacity) typedEmbeddings.remove(typedEmbeddings.keySet().iterator().next());
        }
        return result;
    }

    /** submit_v41: validate the whole request, then submit all images as one scheduler request. */
    private CompletableFuture<List<MockVitImage.Result>> acquireTyped(EngineRpcService.MultimodalInputsPB inputs, long budget) {
        if (closed) return CompletableFuture.failedFuture(Status.UNAVAILABLE.asRuntimeException());
        var work = MockV41Image.validate(inputs, v41, maxImageBytes);
        if (work.size() > gpuScheduler.maxRequestImages())
            throw MockRpcErrors.error(Status.INVALID_ARGUMENT, 902, "[MM_WRONG_FORMAT_ERROR] V4.1 request exceeds image-count limit");
        if (work.isEmpty()) return CompletableFuture.completedFuture(List.of());
        var items = work.stream().map(item -> new MockVitScheduler.Item(() -> forwardTyped(item), 1, null)).toList();
        var submitted = gpuScheduler.submitItems(items);
        var result = submitted.copy().orTimeout(Math.max(0, Math.min(budget, embeddingTimeoutMs)), TimeUnit.MILLISECONDS)
                .exceptionallyCompose(error -> CompletableFuture.failedFuture(cause(error) instanceof java.util.concurrent.TimeoutException
                        ? MockRpcErrors.error(Status.DEADLINE_EXCEEDED, 903, "[MM_PROCESS_ERROR] ViT embedding wait timed out") : cause(error)));
        result.whenComplete((value, error) -> {
            if (error != null) gpuScheduler.cancelRequest(submitted);
        });
        return result;
    }

    @Override public void remoteMultimodalEmbedding(EngineRpcService.MultimodalInputsPB inputs,
                                                   StreamObserver<EngineRpcService.MultimodalOutputPB> observer) {
        long budget = timeout(Context.current(), inputs.getTimeoutMs());
        try {
            var acquired = inputs.hasV41Inputs() ? acquireTyped(inputs, budget) : acquireAll(inputs, budget);
            var output = acquired.thenApply(this::output);
            output.whenComplete((value, error) -> { if (output.isCancelled()) acquired.cancel(false); });
            respond(output, observer, budget, inputs.getRequestId());
        }
        catch (RuntimeException error) {
            observer.onError(failed(error));
        }
    }
    private EngineRpcService.MultimodalOutputPB output(List<MockVitImage.Result> results) {
        var data = new ByteArrayOutputStream(); var ids = new ByteArrayOutputStream();
        var output = EngineRpcService.MultimodalOutputPB.newBuilder();
        int rows = 0;
        boolean bf16 = !results.isEmpty() && results.get(0).embedding().getDataType() == EngineRpcService.TensorPB.DataType.BF16;
        for (var result : results) {
            if ((long)data.size() + result.embeddingBytes() > config.maxOutputBytes())
                throw new IllegalArgumentException("aggregate embedding exceeds output byte limit");
            data.writeBytes((bf16 ? result.embedding().getBf16Data() : result.embedding().getFp32Data()).toByteArray());
            ids.writeBytes(result.hashes().getInt32Data().toByteArray());
            output.addSplitSize(result.rows()); rows = Math.addExact(rows, result.rows());
        }
        var tensor = EngineRpcService.TensorPB.newBuilder().addShape(rows)
                .addShape(bf16 ? v41.hidden() : config.hiddenSize());
        if (bf16) tensor.setDataType(EngineRpcService.TensorPB.DataType.BF16).setBf16Data(ByteString.copyFrom(data.toByteArray()));
        else tensor.setDataType(EngineRpcService.TensorPB.DataType.FP32).setFp32Data(ByteString.copyFrom(data.toByteArray()));
        return output.setMultimodalEmbedding(tensor)
                .setMultimodalFeatureHash(EngineRpcService.TensorPB.newBuilder()
                        .setDataType(EngineRpcService.TensorPB.DataType.INT32).addShape(rows)
                        .setInt32Data(ByteString.copyFrom(ids.toByteArray()))).build();
    }

    @Override public void getMultimodalHashes(EngineRpcService.MultimodalHashRequestPB request,
                                             StreamObserver<EngineRpcService.MultimodalHashResponsePB> observer) {
        long budget = timeout(Context.current(), request.getTimeoutMs());
        try {
            if (request.getKeysCount() > 256 || request.getKeysList().stream().anyMatch(k -> k.isEmpty() || k.codePointCount(0, k.length()) > 4096))
                throw new IllegalArgumentException("invalid multimodal cache keys");
            if (budget <= 0) throw new java.util.concurrent.CompletionException(new java.util.concurrent.TimeoutException("ViT hash acquisition timed out"));
            var base = new ArrayList<EngineRpcService.MultimodalHashEntryPB>();
            var missing = new LinkedHashSet<String>();
            long rows = 0;
            for (String key : request.getKeysList()) {
                var entry = EngineRpcService.MultimodalHashEntryPB.newBuilder().setKey(key);
                var found = hash(key);
                if (found != null) {
                    entry.setHashHit(true).setEntryGeneration(found.generation()).addSplitSize(found.rows()).setFeatureHashes(found.hashes().getInt32Data());
                    rows += found.rows();
                } else missing.add(key);
                String tier = embeddingCache.tier(key);
                if (tier != null) entry.setEmbeddingHit(true).setEmbeddingTier(tier);
                base.add(entry.build());
            }
            checkMetadataRows(rows);
            if (missing.isEmpty() || !request.hasInputs()) {
                observer.onNext(EngineRpcService.MultimodalHashResponsePB.newBuilder().setWorkerInstance(instance).addAllEntries(base).build());
                observer.onCompleted(); return;
            }
            validate(request.getInputs());
            var byKey = new LinkedHashMap<String, EngineRpcService.MultimodalInputPB>();
            for (var image : request.getInputs().getMultimodalInputsList()) {
                if (image.getMultimodalTensor().getSerializedSize() != 0) throw new IllegalArgumentException("hash submission requires URL inputs without tensors");
                byKey.put(MockVitImage.key(image), image);
            }
            if (!byKey.keySet().containsAll(missing) || !new java.util.HashSet<>(request.getKeysList()).containsAll(byKey.keySet()))
                throw new IllegalArgumentException("multimodal inputs do not match requested cache keys");
            var submitted = request.getInputs().toBuilder().clearMultimodalInputs();
            missing.forEach(key -> submitted.addMultimodalInputs(byKey.get(key)));
            var acquired = acquireAll(submitted.build(), budget);
            var response = acquired.thenApply(values -> {
                var completed = new LinkedHashMap<String, MockVitImage.Result>();
                int i = 0; for (String key : missing) completed.put(key, values.get(i++));
                long total = base.stream().flatMap(entry -> entry.getSplitSizeList().stream()).mapToLong(Integer::longValue).sum();
                for (var value : completed.values()) total += value.rows();
                checkMetadataRows(total);
                var output = EngineRpcService.MultimodalHashResponsePB.newBuilder().setWorkerInstance(instance);
                for (var entry : base) {
                    var value = completed.get(entry.getKey());
                    if (value == null) output.addEntries(entry);
                    else {
                        var fresh = EngineRpcService.MultimodalHashEntryPB.newBuilder().setKey(entry.getKey()).setHashHit(true)
                                .addSplitSize(value.rows()).setFeatureHashes(value.hashes().getInt32Data());
                        String tier = embeddingCache.tier(entry.getKey());
                        if (tier != null) fresh.setEmbeddingHit(true).setEmbeddingTier(tier);
                        output.addEntries(fresh);
                    }
                }
                return output.build();
            });
            response.whenComplete((value, error) -> { if (response.isCancelled()) acquired.cancel(false); });
            respond(response, observer, budget, request.getInputs().getRequestId());
        } catch (Throwable error) {
            Throwable root = cause(error);
            observer.onError(root instanceof IllegalArgumentException ? Status.INVALID_ARGUMENT.withDescription(root.getMessage()).asRuntimeException()
                    : root instanceof MetadataOverflow ? Status.RESOURCE_EXHAUSTED.withDescription(root.getMessage()).asRuntimeException() : failed(root));
        }
    }
    private static final class MetadataOverflow extends RuntimeException {
        MetadataOverflow() { super("multimodal metadata response exceeds row limit"); }
    }
    private static void checkMetadataRows(long rows) {
        if (rows > 1048576) throw new MetadataOverflow();
    }

    @Override public void asyncSubmitEmbedding(EngineRpcService.MultimodalInputsPB inputs,
                                               StreamObserver<EngineRpcService.EmptyPB> observer) {
        try {
            validate(inputs);
            claim(inputs.getMultimodalInputsList(), inputs.getRequestId(), null, false);
            observer.onNext(EngineRpcService.EmptyPB.getDefaultInstance()); observer.onCompleted();
        } catch (Throwable error) {
            Throwable root = cause(error);
            observer.onError(failed(root));
        }
    }
    @Override public void waitGreenNetVerdict(EngineRpcService.MultimodalInputsPB inputs,
                                              StreamObserver<EngineRpcService.EmptyPB> observer) {
        observer.onNext(EngineRpcService.EmptyPB.getDefaultInstance()); observer.onCompleted();
    }
    @Override public void releaseRdmaLease(EngineRpcService.ReleaseLeasePB request,
                                           StreamObserver<EngineRpcService.EmptyPB> observer) {
        if (request.getLeaseIdCount() != 0) {
            observer.onError(Status.FAILED_PRECONDITION.withDescription("inline Mock ViT owns no RDMA leases").asRuntimeException()); return;
        }
        observer.onNext(EngineRpcService.EmptyPB.getDefaultInstance()); observer.onCompleted();
    }
    @Override public void getWorkerStatus(EngineRpcService.StatusVersionPB request,
                                          StreamObserver<EngineRpcService.WorkerStatusPB> observer) {
        observer.onNext(EngineRpcService.WorkerStatusPB.newBuilder().setRole("VIT").setStatusVersion(1).setAlive(alive.getAsBoolean()).build());
        observer.onCompleted();
    }
    @Override public synchronized void getCacheStatus(EngineRpcService.CacheVersionPB request,
                                                      StreamObserver<EngineRpcService.CacheStatusPB> observer) {
        var output = EngineRpcService.CacheStatusPB.newBuilder();
        if (request.getNeedCacheKeys()) output.setMultimodalCache(cacheDirectory(100000));
        if (output.build().getSerializedSize() > (16 << 20)) {
            observer.onError(Status.RESOURCE_EXHAUSTED.withDescription("ViT directory exceeds 16 MiB").asRuntimeException()); return;
        }
        observer.onNext(output.build()); observer.onCompleted();
    }
    /** get_mm_cache_keys: recent hashes first, then GPU/CPU residency, one union limit. */
    synchronized EngineRpcService.MultimodalCacheStatusPB cacheDirectory(int limit) {
        var recent = hashCache.keys(limit);
        var tiers = new LinkedHashMap<String, String>();
        for (String key : embeddingCache.keys(MockVitEmbeddingCache.Device.GPU, limit)) tiers.put(key, "gpu");
        for (String key : embeddingCache.keys(MockVitEmbeddingCache.Device.CPU, limit)) {
            if (tiers.size() >= limit) break;
            tiers.putIfAbsent(key, "cpu");
        }
        var selected = new java.util.HashSet<>(recent);
        for (String key : tiers.keySet()) {
            if (selected.size() >= limit) break;
            selected.add(key);
        }
        var directory = EngineRpcService.MultimodalCacheStatusPB.newBuilder().setWorkerInstance(instance).addAllKeys(recent);
        tiers.forEach((key, tier) -> {
            if (selected.contains(key)) {
                if (tier.equals("gpu")) directory.addGpuEmbeddingKeys(key);
                else directory.addCpuEmbeddingKeys(key);
            }
        });
        return directory.build();
    }

    public Map<String, Long> snapshot() {
        var result = new LinkedHashMap<String, Long>(embeddingCache.stats());
        result.putAll(hashCache.stats());
        result.put("url_cache_items", preprocessor.mediaCacheItems());
        result.put("gpu_queued", (long)gpuScheduler.queued()); result.put("gpu_batches", gpuScheduler.batches());
        result.put("pending", (long)pending.size()); result.put("active", (long)executor.getActiveCount());
        result.put("queued", (long)executor.getQueue().size()); result.put("computations", computations.get());
        return result;
    }
    synchronized Map<String, Object> effectiveConfiguration() {
        return Map.of("config", effectiveConfig,
                "simulation", Map.of("hidden_size", config.hiddenSize(), "forward_ms", gpuScheduler.forwardMs(),
                        "max_output_bytes", config.maxOutputBytes()),
                "model", Map.of("V41_IMAGE_ENCODE_CACHE_ITEMS", typedCacheCapacity),
                "limitations", List.of("logical GPU/CPU pools; immutable JVM payloads", "no physical DMA/RDMA or workspace/OOM model", "work estimates require a matching model estimator"));
    }
    @Override public void close() {
        synchronized (this) {
            closed = true;
            executor.shutdown();
            for (Runnable task : executor.getQueue()) {
                if (executor.remove(task) && task instanceof java.util.concurrent.Future<?> queued) queued.cancel(false);
            }
        }
        try {
            while (!executor.awaitTermination(1, TimeUnit.SECONDS)) { /* running forwards retain ownership */ }
        } catch (InterruptedException interrupted) { Thread.currentThread().interrupt(); }
        preprocessor.close();
        gpuScheduler.close();
        synchronized (this) {
            pending.values().forEach(value -> value.completeExceptionally(Status.UNAVAILABLE.asRuntimeException())); pending.clear();
            tasks.values().forEach(task -> task.worker.cancel(true)); tasks.clear(); owners.clear(); typedEmbeddings.clear();
        }
        embeddingCache.close(); hashCache.close();
    }
}
