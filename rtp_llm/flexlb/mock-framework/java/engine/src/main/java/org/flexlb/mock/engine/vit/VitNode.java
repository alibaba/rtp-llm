package org.flexlb.mock.engine.vit;

import io.grpc.Server;
import io.grpc.netty.NettyServerBuilder;
import java.util.*;
import java.util.concurrent.TimeUnit;
import org.flexlb.mock.protocol.*;
import org.flexlb.mock.engine.EngineMain.Address;

/** Standalone vision instance serving Prefill over gRPC. */
public final class VitNode implements AutoCloseable {
    public record CaptureRead(long start, String owner, long expires, long offset, int size, String signature) {}
    public record CaptureChunk(String owner, long start, String name, long bytes, String sha256, long offset,
                               String chunk_sha256, String data_base64) {}

    private final VitConfig config;
    private final VitService service;
    private final Address address=Address.local();
    private Server rpc;
    private ControlServer api;
    private org.flexlb.mock.metrics.MetricsOutput metrics;
    private volatile boolean closed;
    private VitNode(VitConfig config) { this.config=config;service=new VitService(config); }
    public static VitNode start(VitConfig config,int rpcPort,int controlPort) throws Exception {
        var node=new VitNode(config);
        try {
            node.rpc=NettyServerBuilder.forAddress(node.address.listener(rpcPort)).addService(node.service).build().start();
            node.metrics=org.flexlb.mock.metrics.MetricsOutput.start(node.service::collectMetrics);
            node.api=new ControlServer(node.address.listener(controlPort)).health(node::checkHealthy);
            var capture = node.service.geometryCapture();
            if (capture != null) node.api.route("vit/capture/read", CaptureRead.class, CaptureChunk.class, capture::read);
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
}
