package org.flexlb.mock.client;

import com.fasterxml.jackson.databind.ObjectMapper;
import io.grpc.*;
import io.grpc.netty.NettyServerBuilder;
import io.grpc.stub.StreamObserver;
import java.net.*;
import java.nio.file.*;
import java.time.Duration;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import org.flexlb.engine.grpc.EngineRpcService.*;
import org.flexlb.engine.grpc.RpcServiceGrpc;
import org.flexlb.mock.engine.config.RtpConfig;
import org.flexlb.mock.engine.execution.LLMMockEngine;
import org.flexlb.mock.engine.scheduler.*;
import org.flexlb.mock.engine.vit.*;
import org.flexlb.schedule.grpc.FlexlbScheduleProtocol.*;
import org.flexlb.schedule.grpc.FlexlbServiceGrpc;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import static org.junit.jupiter.api.Assertions.*;

class NativeVisionTest {
    @TempDir Path directory;
    com.fasterxml.jackson.databind.JsonNode config(String role) {
        var config=role.equals("vit")?Map.of("patch_size", 14, "downsample_ratio", 3, "min_pixels", 0, "max_image_tokens", 1024, "hidden_size", 4,"mock.vit.skip.download",true,"mm_cache_gpu_max_bytes",1024,"mock.compute",Map.of("forward",Map.of("expression","5")))
            :Map.of("mock.scheduler.mode","group","image_token_id",1000,"seq_size_per_block",2,"max_seq_len",128,"test_block_num",32,"reserve_block_ratio",0,"reuse_cache",true,"mock.compute",Map.of("forward",Map.of("expression","1")));
        return new ObjectMapper().valueToTree(Map.of("id", role + "-0", "component_role", role,
            "component", role.equals("vit") ? Map.of("config", config) : Map.of("engine", "rtp", "config", config)));
    }
    FlexlbServerStatusPB server(String role, Map<String, Object> endpoints) {
        var address=endpoints.get("rpc").toString().split(":");return FlexlbServerStatusPB.newBuilder().setRole(role.toUpperCase(Locale.ROOT)).setServerIp(address[0]).setGrpcPort(Integer.parseInt(address[1])).setWorkerGeneration(1).build();
    }
    @Test void nativeSingleAndMasterBatchRequestsTraverseVPD() throws Exception {
        var prefillConfig = config("prefill");
        ((com.fasterxml.jackson.databind.node.ObjectNode) prefillConfig.path("component").path("config"))
            .put("seq_size_per_block", 1).put("prefill_cp_size", 4);
        try (var v = VitNode.start(VitConfig.load(config("vit")), 0, 0);
             var p = LLMMockEngine.start(RtpConfig.load(prefillConfig));
             var d = LLMMockEngine.start(RtpConfig.load(config("decode")))) {
            var routes = List.of(server("vit", v.endpoints()), server("prefill", p.endpoints()), server("decode", d.endpoints()));
            var stages=new AtomicInteger();var epoch=new AtomicInteger(1);var rejectedPin=new AtomicBoolean();var pChannel=ManagedChannelBuilder.forTarget(p.rpcAddress()).usePlaintext().build();
            var routeKeys = new ConcurrentHashMap<Long, List<Long>>();
            var master=NettyServerBuilder.forAddress(new InetSocketAddress("127.0.0.1",0)).addService(new FlexlbServiceGrpc.FlexlbServiceImplBase() {
                @Override public void schedule(FlexlbScheduleRequestPB request,StreamObserver<FlexlbScheduleResponsePB> response) {
                    try {
                        if(request.getVitRouteOnly()) { stages.incrementAndGet();response.onNext(FlexlbScheduleResponsePB.newBuilder().setSuccess(true).setCode(200).addServerStatus(routes.get(0).toBuilder().setWorkerGeneration(epoch.get())).build()); }
                        else {
                            assertTrue(request.hasSelectedVit());assertEquals(epoch.get(),request.getSelectedVit().getWorkerGeneration());stages.incrementAndGet();
                            assertEquals(4, request.getCacheKeyBlockSize());
                            assertFalse(request.getBlockCacheKeysList().isEmpty());
                            var previous = routeKeys.putIfAbsent(request.getRequestId(), request.getBlockCacheKeysList());
                            if (previous != null) assertEquals(previous, request.getBlockCacheKeysList());
                            if(request.getRequestId()==1 && rejectedPin.compareAndSet(false,true)) {
                                epoch.incrementAndGet();response.onNext(FlexlbScheduleResponsePB.newBuilder().setSuccess(false).setCode(8408).build());response.onCompleted();return;
                            }
                            var input=GenerateInputPB.parseFrom(request.getGenerateInput());assertNotEquals(0,input.getTokenIds(1));
                            boolean batch=request.getRequestId()==2;
                            if(batch) {
                                var routed=input.toBuilder();routed.getGenerateConfigBuilder().clearRoleAddrs();
                                for(var route:routes) routed.getGenerateConfigBuilder().addRoleAddrs(RoleAddrPB.newBuilder().setRole(RoleAddrPB.RoleType.valueOf(route.getRole())).setIp(route.getServerIp()).setGrpcPort(route.getGrpcPort()));
                                var reply=RpcServiceGrpc.newBlockingStub(pChannel).enqueueBatch(EnqueueBatchRequestPB.newBuilder().setBatchId(2).addDpSlots(EnqueueBatchDpSlotPB.newBuilder()
                                    .setDpRank(0).addRequests(EnqueueBatchExternalInputPB.newBuilder().setInput(routed))).build());
                                assertEquals(0,reply.getErrorsCount());
                            }
                            response.onNext(FlexlbScheduleResponsePB.newBuilder().setSuccess(true).setCode(200).setEnqueuedByMaster(batch).addAllServerStatus(routes).build());
                        }
                        response.onCompleted();
                    } catch(Exception error) { response.onError(Status.INTERNAL.withCause(error).asRuntimeException()); }
                }
            }).build().start();
            try(var client=new NativeClient(() -> List.of("127.0.0.1:"+master.getPort()), event -> {}, 1000, 0L, 4)) {
                for(long id:List.of(1L,2L,3L)) {
                    var input=GenerateInputPB.newBuilder().setRequestId(id).addAllTokenIds(id == 3 ? List.of(10,1000) : List.of(10,1000,20))
                        .addMultimodalInputs(MultimodalInputPB.newBuilder().setMultimodalType(1).setMultimodalUrl("same-image")
                            .setMmPreprocessConfig(MMPreprocessConfigPB.newBuilder().setWidth(56).setHeight(28)))
                        .setGenerateConfig(GenerateConfigPB.newBuilder().setMaxNewTokens(3).setIgnoreEos(true).setReturnIncremental(true).setCanUsePdSeparation(true).setReuseCache(true)).build();
                    var request=FlexlbScheduleRequestPB.newBuilder().setRequestId(id).setSeqLen(input.getTokenIdsCount()).setMaxNewTokens(3).setCacheKeyBlockSize(1).setGenerateInput(input.toByteString()).build();
                    var result=client.send(request,Duration.ofSeconds(8)).completion().toCompletableFuture().get(10,TimeUnit.SECONDS);
                    assertEquals(NativeClient.Outcome.success,result.outcome(),result.toString());assertEquals(3,result.outputTokens());
                    var cache = RpcServiceGrpc.newBlockingStub(pChannel).getCacheStatus(CacheVersionPB.newBuilder()
                        .setNeedCacheKeys(true).setLatestCacheVersion(-1).build());
                    assertEquals(4, cache.getBlockSize());
                    assertTrue(cache.getCacheKeysMap().keySet().containsAll(routeKeys.get(id)));
                }
                assertEquals(8,stages.get());
            } finally { master.shutdownNow().awaitTermination(2,TimeUnit.SECONDS);pChannel.shutdownNow().awaitTermination(2,TimeUnit.SECONDS); }
        }
    }
}
