package org.flexlb.mock.engine.vit;

import org.flexlb.mock.protocol.MultimodalProtocol;
import com.google.protobuf.ByteString;
import io.grpc.*;
import io.grpc.netty.NettyServerBuilder;
import io.grpc.stub.*;
import java.io.ByteArrayOutputStream;
import java.util.*;
import java.util.concurrent.*;
import org.flexlb.engine.grpc.EngineRpcService.*;
import org.flexlb.engine.grpc.MultimodalRpcServiceGrpc;
import org.flexlb.mock.protocol.*;
import org.flexlb.mock.engine.EngineMain.Address;

/**
 * Starts and stops a standalone ViT instance and exposes its RPC and control endpoints.
 */
public final class VitNode implements AutoCloseable {
    private final VitConfig config;
    private final RpcService service;
    private final Address address=Address.local();
    private Server rpc;
    private ControlServer api;
    private org.flexlb.mock.metrics.MetricsOutput metrics;
    private volatile boolean closed;
    private VitNode(VitConfig config) { this.config=config;service=new RpcService(config, new VitMetrics()); }
    public static VitNode start(VitConfig config,int rpcPort,int controlPort) throws Exception {
        var node=new VitNode(config);
        try {
            node.rpc=NettyServerBuilder.forAddress(node.address.listener(rpcPort)).addService(node.service).build().start();
            node.metrics=org.flexlb.mock.metrics.MetricsOutput.start(node.service::collectMetrics);
            node.api=new ControlServer(node.address.listener(controlPort)).health(node::checkHealthy);
            if(node.metrics.prometheus()) node.api.metrics(node.service::metrics);
            node.api.start();return node;
        } catch(Exception | Error error) {
            try { node.close(); } catch(RuntimeException cleanup) { error.addSuppressed(cleanup); }
            throw error;
        }
    }
    public Map<String,Object> endpoints() {
        return Map.of("id",config.id(),"role","vit","rpc",address.rpc(rpc.getPort()),"control",address.http(api.port()),
            "metrics",metrics.prometheus()?address.http(api.port())+"/metrics":"","metrics_backend",metrics.prometheus()?"prometheus":"kmonitor","metadata_only",true,"supported_rpc",List.of("GetWorkerStatus","GetCacheStatus","RemoteMultimodalEmbedding","GetMultimodalHashes","AsyncSubmitEmbedding"));
    }
    public void checkHealthy() { if(closed || rpc == null || rpc.isShutdown()) throw new IllegalStateException("V stopped");service.checkHealthy(); }
    @Override public synchronized void close() {
        if(closed) return;closed=true;
        try { service.close(); }
        finally {
            try { if(metrics!=null) metrics.close(); }
            finally {
                try { if(api!=null) api.close(); }
                finally {
                    if(rpc!=null) {
                        rpc.shutdownNow();
                        try { rpc.awaitTermination(3,TimeUnit.SECONDS); }
                        catch(InterruptedException error) { Thread.currentThread().interrupt(); }
                    }
                }
            }
        }
    }

    /**
     * Serves native multimodal RPCs and owns their workers and pending replies.
     */
    static final class RpcService extends MultimodalRpcServiceGrpc.MultimodalRpcServiceImplBase implements AutoCloseable {
        private final VitConfig config;
        private final VitMetrics metrics;
        private final VitCache cache;
        private final VitScheduler scheduler;
        private final VitPreparation preparation;
        private final String instance=UUID.randomUUID().toString();
        private final ExecutorService replies=Executors.newVirtualThreadPerTaskExecutor();

        private volatile boolean closed;

        RpcService(VitConfig config, VitMetrics metrics) {
            this.config = config;
            this.metrics = metrics;
            cache = new VitCache(config, metrics);
            scheduler = new VitScheduler(config, metrics);
            preparation = new VitPreparation(config, cache, scheduler, metrics);
        }

        @Override public void remoteMultimodalEmbedding(MultimodalInputsPB input,StreamObserver<MultimodalOutputPB> response) {
            var context=Context.current(); reply(response,()->output(compute(input,context), input.getAcceptMockFeatureSeeds()));
        }

        @Override public void getMultimodalHashes(MultimodalHashRequestPB input,StreamObserver<MultimodalHashResponsePB> response) {
            var context=Context.current(); reply(response,()-> {
                if(input.getKeysList().stream().anyMatch(String::isBlank)) throw MultimodalProtocol.invalid("Empty hash query key");
                Map<String,VitGeometry.Image> computed=new HashMap<>();
                if(input.getTimeoutMs()<0) throw MultimodalProtocol.invalid("Invalid hash request timeout");
                if(input.hasInputs()) {
                    var media=input.getInputs();
                    VitGeometry.validate(media,config.skipDownload());
                    var requested=new LinkedHashSet<>(input.getKeysList());
                    var byKey=new LinkedHashMap<String,MultimodalInputPB>();
                    for(var image:media.getMultimodalInputsList()) {
                        String key=MultimodalProtocol.key(image);
                        if(!requested.contains(key)) throw MultimodalProtocol.invalid("Hash submission inputs differ from requested keys");
                        byKey.put(key,image);
                    }
                    var missing=media.toBuilder().clearMultimodalInputs();
                    for(var key:requested) {
                        var hit=cache.hash(key);
                        if(hit!=null) computed.put(key,hit);
                        else if(byKey.containsKey(key)) missing.addMultimodalInputs(byKey.get(key));
                        else throw MultimodalProtocol.invalid("Missing image input for hash acquisition");
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
                    if (image != null) {
                        long seed = MultimodalProtocol.featureSeed(image.featureIdentity());
                        entry.setHashHit(true).setEntryGeneration(instance+":"+key).addSplitSize(image.rows());
                        if (input.getAcceptMockFeatureSeeds()) entry.setMockFeatureSeed(seed);
                        else entry.setFeatureHashes(MultimodalProtocol.expandFeatureHashes(seed, image.rows()));
                    }
                    result.addEntries(entry);
                }
                return result.build();
            });
        }

        @Override public void asyncSubmitEmbedding(MultimodalInputsPB input,StreamObserver<EmptyPB> response) {
            reply(response,()-> {
                List<VitCache.Claim> claims;
                try { claims=submit(input, false); }
                catch(RuntimeException error) { metrics.error(error,null);throw error; }
                CompletableFuture.allOf(claims.stream().map(claim->claim.result().toCompletableFuture()).toArray(CompletableFuture[]::new))
                    .whenComplete((value,error) -> claims.forEach(VitCache.Claim::close));
                return EmptyPB.getDefaultInstance();
            });
        }

        @Override public void getWorkerStatus(StatusVersionPB input,StreamObserver<WorkerStatusPB> response) {
            try { scheduler.checkHealthy();response.onNext(WorkerStatusPB.newBuilder().setRole("VIT").setAlive(true).setStatusVersion(1).setMmInflightInputs(com.google.protobuf.Int64Value.of(preparation.inflight())).build());response.onCompleted(); }
            catch(RuntimeException error) { response.onError(Status.UNAVAILABLE.withCause(error).asRuntimeException()); }
        }

        @Override public void getCacheStatus(CacheVersionPB input,StreamObserver<CacheStatusPB> response) {
            var result=CacheStatusPB.newBuilder();
            if(input.getNeedCacheKeys()) result.setMultimodalCache(MultimodalCacheStatusPB.newBuilder().setWorkerInstance(instance)
                .addAllKeys(cache.keys()).addAllGpuEmbeddingKeys(cache.deviceKeys()).addAllCpuEmbeddingKeys(cache.hostKeys()));
            response.onNext(result.build());response.onCompleted();
        }

        @Override public void close() { closed=true;scheduler.close();preparation.close();cache.close();replies.shutdownNow(); }

        java.util.List<io.prometheus.client.Collector.MetricFamilySamples> collectMetrics() { return metrics.collect(scheduler,cache); }

        String metrics() { return metrics.scrape(scheduler,cache); }

        void checkHealthy() { scheduler.checkHealthy(); }

        private List<VitGeometry.Image> compute(MultimodalInputsPB input, Context context) throws Exception {
            long started = System.nanoTime();
            String status="error";
            VitCache.Claim waiting=null;
            List<VitCache.Claim> claims=List.of();
            try {
                claims=submit(input, true);
                long budget=TimeUnit.MILLISECONDS.toNanos(input.getTimeoutMs()>0?input.getTimeoutMs():config.timeoutMs());
                if(context.getDeadline()!=null) budget=Math.min(budget,context.getDeadline().timeRemaining(TimeUnit.NANOSECONDS));
                long end=System.nanoTime()+Math.max(0,budget);
                var result=new ArrayList<VitGeometry.Image>();
                for(var claim:claims) {
                    waiting=claim;
                    var completion=claim.result().toCompletableFuture();
                    while(true) {
                        if(context.isCancelled()) throw Status.CANCELLED.withDescription("V result wait cancelled").asRuntimeException();
                        long left=end-System.nanoTime();if(left<=0) throw Status.DEADLINE_EXCEEDED.withDescription("V request timed out").asRuntimeException();
                        try { result.add(completion.get(Math.min(left,TimeUnit.MILLISECONDS.toNanos(25)),TimeUnit.NANOSECONDS));break; }
                        catch(TimeoutException retry) { /* A caller's deadline never cancels another caller's shared computation. */ }
                        catch(ExecutionException error) {
                            if(error.getCause() instanceof Exception failure) throw failure;
                            throw error;
                        }
                    }
                }
                long rows=result.stream().mapToLong(VitGeometry.Image::rows).sum();
                metrics.events.observe("py_rtp_vit_embedding_length", rows);
                status="success";
                return result;
            } catch (Exception error) { metrics.error(error,waiting); throw error; }
            finally {
                claims.forEach(VitCache.Claim::close);
                metrics.events.observe("py_rtp_vit_request_rt",(System.nanoTime()-started)/1e6,
                    Map.of("source","get_embedding_result","status",status));
            }
        }

        private List<VitCache.Claim> submit(MultimodalInputsPB input, boolean recordImageCount) {
            VitGeometry.validate(input, config.skipDownload());
            if (recordImageCount) metrics.events.observe("py_rtp_vit_image_count", input.getMultimodalInputsCount());
            if (closed) throw Status.UNAVAILABLE.withDescription("V stopped").asRuntimeException();
            return preparation.submit(input);
        }

        private <T> void reply(StreamObserver<T> response, Callable<T> operation) {
            if(closed) { response.onError(Status.UNAVAILABLE.withDescription("V stopped").asRuntimeException()); return; }
            try { replies.execute(()-> {
                try { response.onNext(operation.call());response.onCompleted(); }
                catch(Exception error) { response.onError(error instanceof StatusRuntimeException?error:Status.INTERNAL.withDescription(error.getMessage()).withCause(error).asRuntimeException()); }
            }); } catch(RejectedExecutionException error) { response.onError(Status.UNAVAILABLE.asRuntimeException()); }
        }

        private MultimodalOutputPB output(List<VitGeometry.Image> images, boolean compact) throws Exception {
            int rows = Math.toIntExact(images.stream().mapToLong(VitGeometry.Image::rows).sum());
            var output = MultimodalOutputPB.newBuilder().setMultimodalEmbedding(
                TensorPB.newBuilder().setDataType(TensorPB.DataType.BF16).addShape(rows).addShape(config.hidden()));
            var hashes = new ByteArrayOutputStream();
            for (var image : images) {
                output.addSplitSize(image.rows());
                long seed = MultimodalProtocol.featureSeed(image.featureIdentity());
                if (compact) output.addMockFeatureSeeds(seed);
                else MultimodalProtocol.expandFeatureHashes(seed, image.rows()).writeTo(hashes);
            }
            if (!compact) output.setMultimodalFeatureHash(TensorPB.newBuilder()
                .setDataType(TensorPB.DataType.INT32).addShape(rows).setInt32Data(ByteString.copyFrom(hashes.toByteArray())));
            return output.build();
        }
    }
}
