package org.flexlb.engine.grpc.client;

import com.google.protobuf.ByteString;
import com.google.protobuf.UnknownFieldSet;
import io.grpc.Status;
import io.grpc.netty.NettyServerBuilder;
import io.grpc.stub.StreamObserver;
import io.netty.channel.nio.NioEventLoopGroup;
import org.flexlb.config.CacheMatchConfiguration;
import org.flexlb.config.ConfigService;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.engine.grpc.RpcServiceGrpc;
import org.flexlb.engine.grpc.core.GrpcChannelFactory;
import org.flexlb.engine.grpc.monitor.GrpcReporter;
import org.flexlb.engine.grpc.nameresolver.EngineAddressResolver;
import org.junit.jupiter.api.Test;

import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.Mockito.mock;

class CacheStatusMessageLimitTest {
    @Test
    void acceptsAboveChannelLimitByDefaultAndWithUnifiedOverride() throws Exception {
        exercise(null, true);
        exercise(256 * 1024 * 1024, true);
    }
    @Test
    void enforcesUnifiedLimit() throws Exception {
        exercise(16 * 1024 * 1024, false);
    }
    private void exercise(Integer limit, boolean accepted) throws Exception {
        var response = EngineRpcService.CacheStatusPB.newBuilder()
                .setUnknownFields(UnknownFieldSet.newBuilder().addField(100,
                        UnknownFieldSet.Field.newBuilder()
                                .addLengthDelimited(ByteString.copyFrom(new byte[17 * 1024 * 1024]))
                                .build()).build()).build();
        var server = NettyServerBuilder.forPort(0).addService(new RpcServiceGrpc.RpcServiceImplBase() {
            @Override
            public void getCacheStatus(EngineRpcService.CacheVersionPB request,
                                       StreamObserver<EngineRpcService.CacheStatusPB> observer) {
                observer.onNext(response);
                observer.onCompleted();
            }
        }).build().start();
        var executor = (ThreadPoolExecutor) Executors.newFixedThreadPool(2);
        var group = new NioEventLoopGroup(1);
        var client = new EngineGrpcClient(mock(EngineAddressResolver.class),
                new GrpcChannelFactory(executor, group, 2000), mock(GrpcReporter.class),
                mock(CacheMatchConfiguration.class));
        try {
            if (limit != null) {
                client.configureEngineGrpc(new ConfigService("{\"requestLifecycle\":{\"request\":{\"timeoutMs\":1000}},"
                        + "\"engineGrpc\":{\"kvCacheMaxInboundMessageBytes\":" + limit + "}}"));
            }
            var result = client.getCacheStatusAsync("127.0.0.1", server.getPort(),
                    EngineRpcService.CacheVersionPB.getDefaultInstance(), 10000);
            if (accepted) {
                assertEquals(response.getSerializedSize(), result.get(15, TimeUnit.SECONDS).getSerializedSize());
            } else {
                var failure = assertThrows(ExecutionException.class, () -> result.get(15, TimeUnit.SECONDS));
                assertEquals(Status.Code.RESOURCE_EXHAUSTED, Status.fromThrowable(failure.getCause()).getCode());
            }
        } finally {
            client.shutdown();
            server.shutdownNow().awaitTermination(5, TimeUnit.SECONDS);
            executor.shutdownNow();
            group.shutdownGracefully(0, 1, TimeUnit.SECONDS).sync();
        }
    }
}
