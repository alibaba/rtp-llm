package org.flexlb.mock.vit;

import org.flexlb.mock.multimodal.MediaInput;

import com.google.protobuf.ByteString;
import io.grpc.*;
import io.grpc.stub.*;
import java.io.ByteArrayOutputStream;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import org.flexlb.engine.grpc.EngineRpcService.*;
import org.flexlb.engine.grpc.MultimodalRpcServiceGrpc;

final class VitService extends MultimodalRpcServiceGrpc.MultimodalRpcServiceImplBase implements AutoCloseable {
    private final VitConfig config;
    private final VitMetrics metrics = new VitMetrics();
    private final VitCache cache;
    private final VitScheduler scheduler;
    private final String instance=UUID.randomUUID().toString();
    private final ExecutorService replies=Executors.newVirtualThreadPerTaskExecutor();
    private final AtomicInteger calls=new AtomicInteger();
    private volatile boolean closed;
    VitService(VitConfig config) { this.config=config;cache=new VitCache(config,metrics);scheduler=new VitScheduler(config,cache,metrics); }
    private List<VitCache.Claim> submit(MultimodalInputsPB input) {
        if(closed) throw Status.UNAVAILABLE.withDescription("V stopped").asRuntimeException();
        var images=VitGeometry.images(input,config.patch(),config.merge());
        if(config.batchSize()>1 && images.size()>config.batchImages()) throw MediaInput.invalid("Image count exceeds V forward budget");
        var claims=new ArrayList<VitCache.Claim>(); var producers=new ArrayList<VitCache.Claim>();
        try {
            for(var image:images) { var claim=cache.acquire(image);claims.add(claim);if(claim.producer()) producers.add(claim); }
            scheduler.submit(producers);
        } catch(RuntimeException error) { producers.forEach(c->cache.complete(c,error));claims.forEach(VitCache::release);throw error; }
        return claims;
    }
    private List<VitGeometry.Image> compute(MultimodalInputsPB input, Context context) throws Exception {
        long started = System.nanoTime();
        metrics.events.increment("py_rtp_vit_qps");
        List<VitCache.Claim> claims;
        try { claims=submit(input); }
        catch (RuntimeException error) { metrics.events.increment("py_rtp_vit_error_qps"); throw error; }
        metrics.events.observe("py_rtp_vit_image_count", claims.size());
        long budget=TimeUnit.MILLISECONDS.toNanos(input.getTimeoutMs()>0?input.getTimeoutMs():120000);
        if(context.getDeadline()!=null) budget=Math.min(budget,context.getDeadline().timeRemaining(TimeUnit.NANOSECONDS));
        long end=System.nanoTime()+Math.max(0,budget);
        try {
        var result=new ArrayList<VitGeometry.Image>();
        for(var claim:claims) {
            while(true) {
                if(context.isCancelled()) throw Status.CANCELLED.withDescription("V result wait cancelled").asRuntimeException();
                long left=end-System.nanoTime();if(left<=0) throw Status.DEADLINE_EXCEEDED.withDescription("V request timed out").asRuntimeException();
                try { result.add(claim.result().get(Math.min(left,TimeUnit.MILLISECONDS.toNanos(25)),TimeUnit.NANOSECONDS));break; }
                catch(TimeoutException retry) { /* A caller's deadline never cancels another caller's shared computation. */ }
                catch(ExecutionException error) { throw Status.fromThrowable(error.getCause()).asRuntimeException(); }
            }
        }
        metrics.events.observe("py_rtp_vit_embedding_length", result.stream().mapToLong(VitGeometry.Image::rows).sum());
        metrics.events.observe("py_rtp_vit_embedding_rt", (System.nanoTime() - started) / 1e6);
        metrics.events.increment("py_rtp_vit_success_qps");
        return result;
        } catch (Exception error) { metrics.events.increment("py_rtp_vit_error_qps"); throw error; }
        finally {
            claims.forEach(VitCache::release);
        }
    }
    private <T> void reply(StreamObserver<T> response, Callable<T> operation) {
        if(closed || calls.incrementAndGet()>config.queueSize()+config.batchSize()) {
            if(!closed) calls.decrementAndGet();response.onError(Status.RESOURCE_EXHAUSTED.withDescription("V RPC capacity full").asRuntimeException());return;
        }
        try { replies.execute(()-> {
            try { response.onNext(operation.call());response.onCompleted(); }
            catch(Exception error) { response.onError(error instanceof StatusRuntimeException?error:Status.INTERNAL.withDescription(error.getMessage()).withCause(error).asRuntimeException()); }
            finally { calls.decrementAndGet(); }
        }); } catch(RejectedExecutionException error) { calls.decrementAndGet();response.onError(Status.UNAVAILABLE.asRuntimeException()); }
    }
    @Override public void remoteMultimodalEmbedding(MultimodalInputsPB input,StreamObserver<MultimodalOutputPB> response) {
        var context=Context.current(); reply(response,()->output(compute(input,context)));
    }
    private MultimodalOutputPB output(List<VitGeometry.Image> images) throws Exception {
        int rows=images.stream().mapToInt(VitGeometry.Image::rows).sum();var hashes=new ByteArrayOutputStream();
        var output=MultimodalOutputPB.newBuilder();
        for(var image:images) { output.addSplitSize(image.rows());VitGeometry.hashes(image.key(),image.rows()).writeTo(hashes); }
        return output.setMultimodalEmbedding(TensorPB.newBuilder().setDataType(TensorPB.DataType.BF16).addShape(rows).addShape(config.hidden()))
            .setMultimodalFeatureHash(TensorPB.newBuilder().setDataType(TensorPB.DataType.INT32).addShape(rows).setInt32Data(ByteString.copyFrom(hashes.toByteArray()))).build();
    }
    @Override public void getMultimodalHashes(MultimodalHashRequestPB input,StreamObserver<MultimodalHashResponsePB> response) {
        var context=Context.current(); reply(response,()-> {
            if(input.getKeysCount()>256 || input.getKeysList().stream().anyMatch(k->k.isBlank() || k.length()>4096)) throw MediaInput.invalid("Oversized hash query");
            Map<String,VitGeometry.Image> computed=new HashMap<>();
            if(input.getTimeoutMs()<0 || input.getTimeoutMs()>3600000) throw MediaInput.invalid("Invalid hash request timeout");
            if(input.hasInputs()) {
                var media=input.getInputs();
                VitGeometry.images(media,config.patch(),config.merge());
                var requested=new LinkedHashSet<>(input.getKeysList());
                var byKey=new LinkedHashMap<String,MultimodalInputPB>();
                for(var image:media.getMultimodalInputsList()) {
                    String key=MediaInput.key(image);
                    if(!requested.contains(key)) throw MediaInput.invalid("Hash submission inputs differ from requested keys");
                    byKey.put(key,image);
                }
                var missing=media.toBuilder().clearMultimodalInputs();
                for(var key:requested) {
                    var hit=cache.hash(key);
                    if(hit!=null) computed.put(key,hit);
                    else if(byKey.containsKey(key)) missing.addMultimodalInputs(byKey.get(key));
                    else throw MediaInput.invalid("Missing image input for hash acquisition");
                }
                if(input.getTimeoutMs()>0 && (media.getTimeoutMs()<=0 || input.getTimeoutMs()<media.getTimeoutMs()))
                    missing.setTimeoutMs(input.getTimeoutMs());
                if(missing.getMultimodalInputsCount()>0) for(var image:compute(missing.build(),context)) computed.put(image.key(),image);
            }
            var result=MultimodalHashResponsePB.newBuilder().setWorkerInstance(instance);
            for(var key:input.getKeysList()) {
                var image=computed.get(key);if(image==null) image=cache.hash(key);
                var entry=MultimodalHashEntryPB.newBuilder().setKey(key);String tier=cache.tier(key);
                entry.setEmbeddingHit(!tier.isEmpty()).setEmbeddingTier(tier);
                if(image!=null) entry.setHashHit(true).setEntryGeneration(instance+":"+key).addSplitSize(image.rows()).setFeatureHashes(VitGeometry.hashes(key,image.rows()));
                result.addEntries(entry);
            }
            return result.build();
        });
    }
    @Override public void asyncSubmitEmbedding(MultimodalInputsPB input,StreamObserver<EmptyPB> response) {
        reply(response,()-> { submit(input);return EmptyPB.getDefaultInstance(); });
    }
    @Override public void getWorkerStatus(StatusVersionPB input,StreamObserver<WorkerStatusPB> response) {
        try { scheduler.checkHealthy();response.onNext(WorkerStatusPB.newBuilder().setRole("VIT").setAlive(true).setStatusVersion(1).build());response.onCompleted(); }
        catch(RuntimeException error) { response.onError(Status.UNAVAILABLE.withCause(error).asRuntimeException()); }
    }
    @Override public void getCacheStatus(CacheVersionPB input,StreamObserver<CacheStatusPB> response) {
        var result=CacheStatusPB.newBuilder();
        if(input.getNeedCacheKeys()) result.setMultimodalCache(MultimodalCacheStatusPB.newBuilder().setWorkerInstance(instance)
            .addAllKeys(cache.keys()).addAllGpuEmbeddingKeys(cache.deviceKeys()).addAllCpuEmbeddingKeys(cache.hostKeys()));
        if(result.build().getSerializedSize()>4*1024*1024) { response.onError(Status.RESOURCE_EXHAUSTED.withDescription("V cache snapshot exceeds RPC limit").asRuntimeException());return; }
        response.onNext(result.build());response.onCompleted();
    }
    String metrics() { return metrics.scrape(scheduler,cache); }
    void checkHealthy() { scheduler.checkHealthy(); }
    @Override public void close() { closed=true;scheduler.close();cache.close();replies.shutdownNow(); }
}
