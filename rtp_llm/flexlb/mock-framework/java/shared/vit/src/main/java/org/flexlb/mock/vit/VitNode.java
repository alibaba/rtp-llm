package org.flexlb.mock.vit;

import io.grpc.Server;
import io.grpc.netty.NettyServerBuilder;
import java.util.*;
import java.util.concurrent.TimeUnit;
import org.flexlb.mock.runtime.*;

/** Backend-independent V instance. The same listeners serve shared JVM and remote P. */
public final class VitNode implements EngineInstance {
    private final VitConfig config;
    private final VitService service;
    private final EngineAddress address=EngineAddress.local();
    private Server rpc;
    private ApiServer api;
    private boolean closed;
    private VitNode(VitConfig config) { this.config=config;service=new VitService(config); }
    public static VitNode start(VitConfig config,int rpcPort,int controlPort) throws Exception {
        var node=new VitNode(config);
        try {
            node.rpc=NettyServerBuilder.forAddress(node.address.listener(rpcPort)).addService(node.service).build().start();
            node.api=new ApiServer(node.address.listener(controlPort)).metrics(node.service::metrics).start();return node;
        } catch(Exception error) { node.close();throw error; }
    }
    @Override public Map<String,Object> endpoints() {
        return Map.of("id",config.id(),"role","vit","rpc",address.rpc(rpc.getPort()),"control",address.http(api.port()),
            "metrics",address.http(api.port())+"/metrics","metadata_only",true,"supported_rpc",List.of("GetWorkerStatus","GetCacheStatus","RemoteMultimodalEmbedding","GetMultimodalHashes","AsyncSubmitEmbedding"));
    }
    @Override public void checkHealthy() { if(closed) throw new IllegalStateException("V closed");service.checkHealthy(); }
    @Override public synchronized void close() {
        if(closed) return;closed=true;service.close();if(api!=null) api.close();
        if(rpc!=null) { rpc.shutdownNow();try { rpc.awaitTermination(3,TimeUnit.SECONDS); } catch(InterruptedException error) { Thread.currentThread().interrupt(); } }
    }
}
