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
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;

/** Main ViT wire contract with deterministic forward, byte-bounded caches and shared computation. */
public final class MockVitService extends MultimodalRpcServiceGrpc.MultimodalRpcServiceImplBase implements AutoCloseable {
    public record Config(int hiddenSize, int concurrency, int queueCapacity, long embeddingCacheBytes,
                         long hashCacheBytes, int maxInputBytes, long maxOutputBytes, long computeDelayMs) {
        public Config {
            if (hiddenSize <= 0 || concurrency <= 0 || queueCapacity < 0 || embeddingCacheBytes < 0
                    || hashCacheBytes < 0 || maxInputBytes <= 0 || maxOutputBytes <= 0 || computeDelayMs < 0)
                throw new IllegalArgumentException("invalid Mock ViT limits");
        }
        public static Config defaults() {
            return new Config(64, 4, 64, 64L << 20, 8L << 20, 16 << 20, 64L << 20, 1);
        }
    }
    final Config config;
    private final MockV41Image.Config v41;
    final String instance = UUID.randomUUID().toString().replace("-", "");
    private final ThreadPoolExecutor executor;
    private final Map<String, CompletableFuture<MockVitImage.Result>> pending = new ConcurrentHashMap<>();
    private final MockVitEmbeddingCache<MockVitImage.Result> embeddingCache;
    private final MockVitHashCache hashCache;
    private final MockVitMediaCache mediaCache;
    private final long embeddingTimeoutMs;
    private final int maxImageBytes;

    private final MockVitScheduler gpuScheduler;
    private final MockVitPreprocessor preprocessor;
    private final Map<String, String> downloadHeaders;
    private final com.fasterxml.jackson.databind.JsonNode effectiveConfig;
    private final Map<CompletableFuture<MockVitImage.Result>, java.util.concurrent.atomic.AtomicInteger> owners = new ConcurrentHashMap<>();
    private final Map<CompletableFuture<MockVitImage.Result>, java.util.concurrent.FutureTask<Void>> tasks = new ConcurrentHashMap<>();
    private volatile boolean closed;
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
        effectiveConfig = document.deepCopy();
        var limits = document.path("roles").path("vit");
        if (config.concurrency() != limits.path("vit_concurrency").intValue()
                || config.queueCapacity() != limits.path("vit_max_queue_size").intValue()
                || config.embeddingCacheBytes() != limits.path("mm_cache_cpu_max_bytes").longValue()
                || config.hashCacheBytes() != limits.path("mm_hash_key_cache_max_bytes").longValue())
            throw new IllegalArgumentException("ViT runtime limits differ from effective config");
        if (MockEngineConfig.vitWorkBudget(limits) != null)
            throw new IllegalArgumentException("configured ViT model has no work estimator; cannot execute a nonnull work_budget");
        embeddingTimeoutMs = limits.path("mm_timeout_ms").longValue();
        maxImageBytes = MockEngineConfig.vitImageBytes(limits);
        preprocessor = new MockVitPreprocessor(limits.path("use_local_preprocess").booleanValue(),
                limits.path("mm_preprocess_max_workers").intValue());
        downloadHeaders = MockEngineConfig.vitDownloadHeaders(limits);
        embeddingCache = new MockVitEmbeddingCache<>(limits.path("mm_cache_gpu_max_bytes").longValue(), config.embeddingCacheBytes());
        hashCache = new MockVitHashCache(config.hashCacheBytes(), limits.path("hash_cache_accounting"));
        mediaCache = new MockVitMediaCache(limits.path("url_cache_item_num").intValue());
        gpuScheduler = new MockVitScheduler(limits.path("mm_max_queue_size").intValue(),
                limits.path("gpu_max_batch_size").intValue(), limits.path("gpu_max_batch_images").longValue(),
                limits.path("gpu_batch_wait_ms").longValue(), config.computeDelayMs(), MockEngineConfig.vitWorkBudget(limits));
    }

    void setSchedulerMetrics(java.util.function.Consumer<Map<String, Number>> reporter) {
        gpuScheduler.setMetrics(reporter);
        preprocessor.setMetrics(reporter);
    }

    private CompletableFuture<MockVitImage.Result> cached(String key, long budgetMs) {
        var read = embeddingCache.claim(key, budgetMs);
        if (read == null) return null;
        var result = new CompletableFuture<MockVitImage.Result>();
        // Cache consumption belongs to the caller, outside shared admission/queue locks.
        Thread.startVirtualThread(() -> {
            try {
                var hit = read.get();
                if (hashCache.consume(key, hit.generation()) == null) hashCache.put(key, hit.value(), hit.generation());
                result.complete(hit.value());
            } catch (Throwable error) { result.completeExceptionally(error); }
        });
        return result;
    }
    private MockVitHashCache.Entry hash(String key) { return hashCache.peek(key); }
    private void remember(String key, MockVitImage.Result result) {
        if (closed || key.startsWith("uncached:")) return;
        String generation = UUID.randomUUID().toString().replace("-", "");
        // Feature hashes are an independent CPU sidecar, not embedding-pool payload.
        var layout = List.of(new MockVitEmbeddingCache.Tensor(result.embedding(), result.embeddingBytes(),
                result.embedding().getDataType() == EngineRpcService.TensorPB.DataType.BF16 ? 2 : 4,
                MockVitEmbeddingCache.Device.GPU));
        embeddingCache.put(key, result, layout, generation);
        hashCache.put(key, result, generation);
    }

    CompletableFuture<MockVitImage.Result> acquire(EngineRpcService.MultimodalInputPB image, long timeoutMs) {
        long mediaTimeout = image.getMmPreprocessConfig().getMmTimeoutMs() > 0
                ? image.getMmPreprocessConfig().getMmTimeoutMs() : embeddingTimeoutMs;
        long preparedAt = System.nanoTime();
        java.util.function.LongSupplier budget = () -> Math.min(timeoutMs, mediaTimeout)
                - TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - preparedAt);
        return acquirePrepared(MockVitImage.key(image), timeoutMs, mediaTimeout, () -> {
            var prepared = preprocessor.submit(() -> MockVitImage.prepare(image, config.hiddenSize(), maxImageBytes,
                    config.maxOutputBytes(), positivePreprocessBudget(budget), downloadHeaders, mediaCache))
                    .get(positivePreprocessBudget(budget), TimeUnit.MILLISECONDS);
            return () -> MockVitImage.forward(prepared);
        });
    }

    private static long positivePreprocessBudget(java.util.function.LongSupplier budget) throws java.util.concurrent.TimeoutException {
        long remaining = budget.getAsLong();
        if (remaining <= 0) throw new java.util.concurrent.TimeoutException("ViT preprocessing deadline exceeded");
        return remaining;
    }

    private CompletableFuture<MockVitImage.Result> acquirePrepared(String key, long timeoutMs, long mediaTimeout,
            java.util.concurrent.Callable<java.util.concurrent.Callable<MockVitImage.Result>> prepare) {
        if (closed) return CompletableFuture.failedFuture(Status.UNAVAILABLE.asRuntimeException());
        if (timeoutMs <= 0) return CompletableFuture.failedFuture(Status.DEADLINE_EXCEEDED.asRuntimeException());
        var result = cached(key, timeoutMs);
        if (result != null) return result;
        return enqueuePrepared(key, timeoutMs, mediaTimeout, prepare);
    }

    private synchronized CompletableFuture<MockVitImage.Result> enqueuePrepared(String key, long timeoutMs, long mediaTimeout,
            java.util.concurrent.Callable<java.util.concurrent.Callable<MockVitImage.Result>> prepare) {
        if (closed) return CompletableFuture.failedFuture(Status.UNAVAILABLE.asRuntimeException());
        var shared = pending.get(key);
        if (shared != null) return shared;
        if (tasks.size() >= (long)config.concurrency() + config.queueCapacity())
            return CompletableFuture.failedFuture(MockRpcErrors.error(Status.RESOURCE_EXHAUSTED, 409, "[CONCURRENCY_LIMIT_ERROR] ViT async admission is full"));
        var created = new CompletableFuture<MockVitImage.Result>();
        var existing = pending.putIfAbsent(key, created);
        if (existing != null) return existing;
        long submittedAt = System.nanoTime();
        try {
            var task = new java.util.concurrent.FutureTask<Void>(() -> {
                try {
                    long remaining = Math.min(timeoutMs, mediaTimeout) - TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - submittedAt);
                    if (remaining <= 0) throw Status.DEADLINE_EXCEEDED.withDescription("ViT async queue deadline exceeded").asRuntimeException();
                    var resident = cached(key, remaining);
                    if (resident != null) { created.complete(resident.get(remaining, TimeUnit.MILLISECONDS)); return; }
                    var forward = prepare.call();
                    remaining = Math.min(timeoutMs, mediaTimeout) - TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - submittedAt);
                    if (remaining <= 0) throw Status.DEADLINE_EXCEEDED.withDescription("ViT preprocessing deadline exceeded").asRuntimeException();
                    MockVitImage.Result computed;
                    var gpuWork = gpuScheduler.submit(forward);
                    try { computed = gpuWork.get(Math.min(remaining, mediaTimeout), TimeUnit.MILLISECONDS); }
                    catch (java.util.concurrent.TimeoutException error) {
                        gpuScheduler.cancelRequest(gpuWork);
                        throw Status.DEADLINE_EXCEEDED.withDescription("ViT GPU queue/forward deadline exceeded").asRuntimeException();
                    }
                    catch (InterruptedException error) { gpuScheduler.cancelRequest(gpuWork); throw error; }
                    computations.incrementAndGet(); remember(key, computed); created.complete(computed);
                } catch (Throwable error) { created.completeExceptionally(error); }
                finally { pending.remove(key, created); tasks.remove(created); }
            }, null);
            tasks.put(created, task);
            executor.execute(task);
        } catch (RejectedExecutionException error) {
            pending.remove(key, created); tasks.remove(created); created.completeExceptionally(Status.RESOURCE_EXHAUSTED
                    .withDescription("Mock ViT forward queue is full").asRuntimeException());
        }
        return created;
    }

    static long timeout(Context context, long requested) {
        long remaining = context.getDeadline() == null ? 120000 : context.getDeadline().timeRemaining(TimeUnit.MILLISECONDS);
        return Math.min(requested > 0 ? requested : 120000, remaining);
    }
    private static Throwable cause(Throwable error) {
        while ((error instanceof CompletionException || error instanceof java.util.concurrent.ExecutionException) && error.getCause() != null) error = error.getCause();
        return error;
    }
    private static <T> void respond(CompletableFuture<T> work, StreamObserver<T> observer, long budget) {
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
            if (error != null && !work.isDone()) work.cancel(false);
            if (context.isCancelled()) return;
            if (error == null) { observer.onNext(value); observer.onCompleted(); }
            else {
                Throwable root = cause(error);
                Status status = root instanceof java.util.concurrent.TimeoutException ? Status.DEADLINE_EXCEEDED
                        : root instanceof IllegalArgumentException ? Status.INVALID_ARGUMENT
                        : Status.fromThrowable(root);
                if (status.getCode() == Status.Code.UNKNOWN) status = Status.INTERNAL;
                observer.onError(root instanceof io.grpc.StatusRuntimeException || root instanceof io.grpc.StatusException
                        ? root : status.withDescription(root.getMessage()).asRuntimeException());
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
    private synchronized CompletableFuture<List<MockVitImage.Result>> acquireAll(EngineRpcService.MultimodalInputsPB inputs, long budget) {
        validate(inputs);
        long misses = inputs.getMultimodalInputsList().stream().map(MockVitImage::key).distinct()
                .filter(key -> !embeddingCache.contains(key) && !pending.containsKey(key)).count();
        if (tasks.size() + misses > (long)config.concurrency() + config.queueCapacity())
            throw MockRpcErrors.error(Status.RESOURCE_EXHAUSTED, 409, "[CONCURRENCY_LIMIT_ERROR] ViT async batch exceeds admission capacity");
        var values = inputs.getMultimodalInputsList().stream().map(image -> acquire(image, budget)).toList();
        values.forEach(value -> owners.computeIfAbsent(value, ignored -> new java.util.concurrent.atomic.AtomicInteger()).incrementAndGet());
        var shared = CompletableFuture.allOf(values.toArray(CompletableFuture[]::new));
        var result = shared.thenApply(ignored -> values.stream().map(CompletableFuture::join).toList());
        result.whenComplete((value, error) -> values.forEach(this::releaseOwner));
        return result;
    }

    private synchronized void releaseOwner(CompletableFuture<MockVitImage.Result> work) {
        var count = owners.get(work);
        if (count == null || count.decrementAndGet() != 0) return;
        owners.remove(work, count);
        var task = tasks.get(work);
        if (task != null && executor.remove(task)) {
            tasks.remove(work, task); pending.values().removeIf(value -> value == work);
            task.cancel(false); work.cancel(false);
        }
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
            if (error != null && gpuScheduler != null) gpuScheduler.cancelRequest(submitted);
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
            respond(output, observer, budget);
        }
        catch (RuntimeException error) {
            observer.onError(error instanceof IllegalArgumentException
                    ? Status.INVALID_ARGUMENT.withDescription(error.getMessage()).asRuntimeException() : error);
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

    @Override public synchronized void getMultimodalHashes(EngineRpcService.MultimodalHashRequestPB request,
                                             StreamObserver<EngineRpcService.MultimodalHashResponsePB> observer) {
        long budget = timeout(Context.current(), request.getTimeoutMs());
        try {
            if (request.getKeysCount() > 256 || request.getKeysList().stream().anyMatch(k -> k.isBlank() || k.length() > 4096))
                throw new IllegalArgumentException("invalid multimodal cache keys");
            var keys = new LinkedHashSet<>(request.getKeysList());
            Map<String, MockVitImage.Result> transientResults = new ConcurrentHashMap<>();
            List<CompletableFuture<?>> work = new ArrayList<>();
            if (request.hasInputs()) {
                validate(request.getInputs());
                Map<String, EngineRpcService.MultimodalInputPB> submitted = new LinkedHashMap<>();
                for (var image : request.getInputs().getMultimodalInputsList()) {
                    String key = MockVitImage.key(image);
                    if (!keys.contains(key)) throw new IllegalArgumentException("image does not match requested cache keys");
                    submitted.put(key, image);
                }
                for (String key : keys) if (hash(key) == null) {
                    var image = submitted.get(key);
                    if (image == null) throw new IllegalArgumentException("missing image for uncached key");
                    var shared = acquire(image, budget);
                    owners.computeIfAbsent(shared, ignored -> new java.util.concurrent.atomic.AtomicInteger()).incrementAndGet();
                    var waiter = shared.thenAccept(value -> transientResults.put(key, value));
                    waiter.whenComplete((value, error) -> releaseOwner(shared));
                    work.add(waiter);
                }
            }
            var responseWork = CompletableFuture.allOf(work.toArray(CompletableFuture[]::new)).thenApply(ignored -> {
                var response = EngineRpcService.MultimodalHashResponsePB.newBuilder().setWorkerInstance(instance);
                for (String key : keys) {
                    var entry = response.addEntriesBuilder().setKey(key);
                    MockVitHashCache.Entry found = hash(key);
                    var computed = transientResults.get(key);
                    if (found != null || computed != null) {
                        entry.setHashHit(true).setEntryGeneration(found == null ? "" : found.generation())
                                .addSplitSize(found == null ? computed.rows() : found.rows())
                                .setFeatureHashes(found == null ? computed.hashes().getInt32Data() : found.hashes().getInt32Data());
                    }
                    synchronized (this) {
                        String tier = embeddingCache.tier(key);
                        if (tier != null) entry.setEmbeddingHit(true).setEmbeddingTier(tier);
                    }
                }
                return response.build();
            });
            responseWork.whenComplete((value, error) -> {
                if (responseWork.isCancelled()) work.forEach(waiter -> waiter.cancel(false));
            });
            respond(responseWork, observer, budget);
        } catch (RuntimeException error) {
            observer.onError(error instanceof IllegalArgumentException
                    ? Status.INVALID_ARGUMENT.withDescription(error.getMessage()).asRuntimeException() : error);
        }
    }

    @Override public void asyncSubmitEmbedding(EngineRpcService.MultimodalInputsPB inputs,
                                               StreamObserver<EngineRpcService.EmptyPB> observer) {
        try {
            var futures = acquireAll(inputs, timeout(Context.current(), inputs.getTimeoutMs()));
            if (futures.isCompletedExceptionally()) futures.join();
            observer.onNext(EngineRpcService.EmptyPB.getDefaultInstance()); observer.onCompleted();
        } catch (Throwable error) {
            Throwable root = cause(error);
            observer.onError((root instanceof IllegalArgumentException ? Status.INVALID_ARGUMENT : Status.fromThrowable(root))
                    .withDescription(root.getMessage()).asRuntimeException());
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
        observer.onNext(EngineRpcService.WorkerStatusPB.newBuilder().setRole("VIT").setStatusVersion(1).setAlive(true).build());
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
        result.put("url_cache_items", mediaCache.size());
        result.put("gpu_queued", (long)gpuScheduler.queued()); result.put("gpu_batches", gpuScheduler.batches());
        result.put("pending", (long)pending.size()); result.put("active", (long)executor.getActiveCount());
        result.put("queued", (long)executor.getQueue().size()); result.put("computations", computations.get());
        return result;
    }
    synchronized Map<String, Object> effectiveConfiguration() {
        return Map.of("config", effectiveConfig,
                "simulation", Map.of("hidden_size", config.hiddenSize(), "forward_ms", config.computeDelayMs(),
                        "max_output_bytes", config.maxOutputBytes()),
                "model", Map.of("V41_IMAGE_ENCODE_CACHE_ITEMS", typedCacheCapacity),
                "limitations", List.of("logical GPU/CPU pools; immutable JVM payloads", "no physical DMA/RDMA or workspace/OOM model", "work estimates require a matching model estimator"));
    }
    @Override public void close() {
        closed = true;
        executor.shutdown();
        for (Runnable task : executor.getQueue()) {
            if (executor.remove(task) && task instanceof java.util.concurrent.Future<?> queued) queued.cancel(false);
        }
        try {
            while (!executor.awaitTermination(1, TimeUnit.SECONDS)) { /* running forwards retain ownership */ }
        } catch (InterruptedException interrupted) { Thread.currentThread().interrupt(); }
        preprocessor.close();
        gpuScheduler.close();
        pending.values().forEach(value -> value.completeExceptionally(Status.UNAVAILABLE.asRuntimeException())); pending.clear();
        tasks.values().forEach(task -> task.cancel(true)); tasks.clear(); owners.clear();
        embeddingCache.close(); hashCache.close(); mediaCache.clear();
        synchronized (this) { typedEmbeddings.clear(); }
    }
}
