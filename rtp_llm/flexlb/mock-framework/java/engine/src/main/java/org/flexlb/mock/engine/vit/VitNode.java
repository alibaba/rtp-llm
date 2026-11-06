package org.flexlb.mock.engine.vit;

import io.grpc.Server;
import io.grpc.netty.NettyServerBuilder;
import java.util.*;
import java.util.concurrent.TimeUnit;
import org.flexlb.mock.protocol.*;
import org.flexlb.mock.engine.EngineMain.Address;

/** Backend-independent V instance. The same listeners serve shared JVM and remote P. */
public final class VitNode implements AutoCloseable {
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
        try { if(metrics!=null) metrics.close(); }
        finally {
            try { service.close(); }
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
