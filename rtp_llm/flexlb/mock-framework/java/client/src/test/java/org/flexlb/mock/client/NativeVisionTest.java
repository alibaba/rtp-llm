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
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicBoolean;
import org.flexlb.engine.grpc.EngineRpcService.*;
import org.flexlb.engine.grpc.RpcServiceGrpc;
import org.flexlb.schedule.grpc.FlexlbScheduleProtocol.*;
import org.flexlb.schedule.grpc.FlexlbServiceGrpc;
import org.flexlb.mock.host.*;
import org.flexlb.mock.api.model.Types.None;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import static org.junit.jupiter.api.Assertions.*;

class NativeVisionTest {
    @TempDir Path directory;
    int port() throws Exception { try(var socket=new ServerSocket(0)) { return socket.getLocalPort(); } }
    EngineHostApi.Instance create(EngineHost host,String role) throws Exception {
        var config=role.equals("vit")?Map.of("hidden_size",4,"mm_cache_gpu_max_bytes",1024,"compute",Map.of("forward",Map.of("expression","5")))
            :Map.of("seq_size_per_block",2,"max_seq_len",128,"test_block_num",32,"reserve_block_ratio",0,"reuse_cache",true,"compute",Map.of("forward",Map.of("expression","1")));
        var path=directory.resolve(role+"-0");Files.createDirectories(path);
        new ObjectMapper().writeValue(path.resolve("config.json").toFile(),Map.of("id",role+"-0","component_role",role,"component",role.equals("vit")?Map.of("config",config):Map.of("engine","rtp","config",config),"environment",Map.of()));
        return host.create(new EngineHostApi.Create(role+"-0",port(),port()));
    }
    FlexlbServerStatusPB server(EngineHostApi.Instance node) {
        var address=node.rpc().split(":");return FlexlbServerStatusPB.newBuilder().setRole(node.role().toUpperCase(Locale.ROOT)).setServerIp(address[0]).setGrpcPort(Integer.parseInt(address[1])).setWorkerGeneration(1).build();
    }
    @Test void nativeSingleAndMasterBatchRequestsTraverseVPDInOneHostWithoutSharedState() throws Exception {
        try(var host=new EngineHost(directory)) {
            var v=create(host,"vit");var p=create(host,"prefill");var d=create(host,"decode");var routes=List.of(server(v),server(p),server(d));
            var stages=new AtomicInteger();var epoch=new AtomicInteger(1);var rejectedPin=new AtomicBoolean();var pChannel=ManagedChannelBuilder.forTarget(p.rpc()).usePlaintext().build();
            var master=NettyServerBuilder.forAddress(new InetSocketAddress("127.0.0.1",0)).addService(new FlexlbServiceGrpc.FlexlbServiceImplBase() {
                @Override public void schedule(FlexlbScheduleRequestPB request,StreamObserver<FlexlbScheduleResponsePB> response) {
                    try {
                        if(request.getVitRouteOnly()) { stages.incrementAndGet();response.onNext(FlexlbScheduleResponsePB.newBuilder().setSuccess(true).setCode(200).addServerStatus(routes.get(0).toBuilder().setWorkerGeneration(epoch.get())).build()); }
                        else {
                            assertTrue(request.hasSelectedVit());assertEquals(epoch.get(),request.getSelectedVit().getWorkerGeneration());stages.incrementAndGet();
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
            try(var client=new NativeClient("127.0.0.1:"+master.getPort())) {
                for(long id:List.of(1L,2L)) {
                    var input=GenerateInputPB.newBuilder().setRequestId(id).addAllTokenIds(List.of(10,0,0,20))
                        .addMultimodalInputs(MultimodalInputPB.newBuilder().setMultimodalType(1).setMultimodalUrl("same-image")
                            .setMmPreprocessConfig(MMPreprocessConfigPB.newBuilder().setWidth(56).setHeight(28)))
                        .setMultimodalTokenLayout(MultimodalTokenLayoutPB.newBuilder().addSpans(MultimodalTokenSpanPB.newBuilder().setOffset(1).setLength(2)))
                        .setGenerateConfig(GenerateConfigPB.newBuilder().setMaxNewTokens(3).setIgnoreEos(true).setReturnIncremental(true).setCanUsePdSeparation(true)).build();
                    var request=FlexlbScheduleRequestPB.newBuilder().setRequestId(id).setSeqLen(4).setMaxNewTokens(3).setGenerateInput(input.toByteString()).build();
                    var result=client.send(request,Duration.ofSeconds(8)).completion().toCompletableFuture().get(10,TimeUnit.SECONDS);
                    assertEquals(NativeClient.Outcome.success,result.outcome(),result.toString());assertEquals(3,result.outputTokens());
                }
                assertEquals(6,stages.get());assertEquals(3,host.status(new None()).size());
                host.stop(new EngineHostApi.Stop("vit-0"));assertEquals(2,host.status(new None()).size());
            } finally { master.shutdownNow().awaitTermination(2,TimeUnit.SECONDS);pChannel.shutdownNow().awaitTermination(2,TimeUnit.SECONDS); }
        }
    }
}
