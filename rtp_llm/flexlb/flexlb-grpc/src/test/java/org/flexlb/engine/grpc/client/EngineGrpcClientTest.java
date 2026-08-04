package org.flexlb.engine.grpc.client;

import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.Server;
import io.grpc.ServerBuilder;
import io.grpc.Status;
import io.grpc.StatusRuntimeException;
import io.grpc.stub.StreamObserver;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.engine.grpc.MultimodalRpcServiceGrpc;
import org.flexlb.engine.grpc.RpcServiceGrpc;
import org.flexlb.engine.grpc.core.GrpcChannelFactory;
import org.flexlb.engine.grpc.core.GrpcTarget;
import org.flexlb.engine.grpc.monitor.GrpcReporter;
import org.flexlb.engine.grpc.monitor.GrpcRuntimeMetrics;
import org.flexlb.engine.grpc.nameresolver.EngineAddressResolver;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class EngineGrpcClientTest {

    @Test
    void returnsWorkerStatusSynchronously() throws Exception {
        Server server = ServerBuilder.forPort(0)
                .addService(new RpcServiceGrpc.RpcServiceImplBase() {
                    @Override
                    public void getWorkerStatus(
                            EngineRpcService.StatusVersionPB request,
                            StreamObserver<EngineRpcService.WorkerStatusPB> responseObserver) {
                        responseObserver.onNext(EngineRpcService.WorkerStatusPB.newBuilder()
                                .setStatusVersion(77)
                                .build());
                        responseObserver.onCompleted();
                    }
                })
                .build()
                .start();
        ManagedChannel channel = ManagedChannelBuilder.forAddress("127.0.0.1", server.getPort())
                .usePlaintext()
                .build();
        EngineGrpcClient client = null;
        try {
            EngineAddressResolver addressResolver = mock(EngineAddressResolver.class);
            GrpcChannelFactory channelFactory = mock(GrpcChannelFactory.class);
            GrpcReporter grpcReporter = mock(GrpcReporter.class);
            GrpcRuntimeMetrics grpcRuntimeMetrics = mock(GrpcRuntimeMetrics.class);
            GrpcRuntimeMetrics.GrpcCallHandle callHandle = mock(GrpcRuntimeMetrics.GrpcCallHandle.class);
            when(grpcRuntimeMetrics.recordCallStarted("engine", "GetWorkerStatus", 1_000)).thenReturn(callHandle);
            when(channelFactory.create(new GrpcTarget("127.0.0.1", server.getPort()))).thenReturn(channel);
            client = new EngineGrpcClient(
                    addressResolver, channelFactory, grpcReporter, grpcRuntimeMetrics);

            EngineRpcService.WorkerStatusPB response = client.getWorkerStatus(
                    "127.0.0.1",
                    server.getPort(),
                    EngineRpcService.StatusVersionPB.getDefaultInstance(),
                    1_000);

            org.junit.jupiter.api.Assertions.assertEquals(77, response.getStatusVersion());
            verify(grpcRuntimeMetrics).recordCallStarted("engine", "GetWorkerStatus", 1_000);
            verify(grpcRuntimeMetrics).recordCallCompleted(callHandle);
        } finally {
            close(client, server, channel);
        }
    }

    @Test
    void returnsCacheStatusSynchronously() throws Exception {
        Server server = ServerBuilder.forPort(0)
                .addService(new RpcServiceGrpc.RpcServiceImplBase() {
                    @Override
                    public void getCacheStatus(
                            EngineRpcService.CacheVersionPB request,
                            StreamObserver<EngineRpcService.CacheStatusPB> responseObserver) {
                        responseObserver.onNext(EngineRpcService.CacheStatusPB.newBuilder()
                                .setVersion(33)
                                .build());
                        responseObserver.onCompleted();
                    }
                })
                .build()
                .start();
        ManagedChannel channel = ManagedChannelBuilder.forAddress("127.0.0.1", server.getPort())
                .usePlaintext()
                .build();
        EngineGrpcClient client = null;
        try {
            EngineAddressResolver addressResolver = mock(EngineAddressResolver.class);
            GrpcChannelFactory channelFactory = mock(GrpcChannelFactory.class);
            GrpcReporter grpcReporter = mock(GrpcReporter.class);
            when(channelFactory.create(new GrpcTarget("127.0.0.1", server.getPort()))).thenReturn(channel);
            client = new EngineGrpcClient(
                    addressResolver, channelFactory, grpcReporter, mock(GrpcRuntimeMetrics.class));

            EngineRpcService.CacheStatusPB response = client.getCacheStatus(
                    "127.0.0.1",
                    server.getPort(),
                    EngineRpcService.CacheVersionPB.getDefaultInstance(),
                    1_000);

            org.junit.jupiter.api.Assertions.assertEquals(33, response.getVersion());
        } finally {
            close(client, server, channel);
        }
    }

    @Test
    void returnsCacheGroupMetadataThroughMono() throws Exception {
        Server server = ServerBuilder.forPort(0)
                .addService(new RpcServiceGrpc.RpcServiceImplBase() {
                    @Override
                    public void getKvCacheGroupsMetadata(
                            EngineRpcService.KvCacheGroupsRequestPB request,
                            StreamObserver<EngineRpcService.KvCacheGroupListPB> responseObserver) {
                        responseObserver.onNext(EngineRpcService.KvCacheGroupListPB.newBuilder()
                                .addItems(EngineRpcService.KvCacheGroupMetadataPB.getDefaultInstance())
                                .build());
                        responseObserver.onCompleted();
                    }
                })
                .build()
                .start();
        ManagedChannel channel = ManagedChannelBuilder.forAddress("127.0.0.1", server.getPort())
                .usePlaintext()
                .build();
        EngineGrpcClient client = null;
        try {
            EngineAddressResolver addressResolver = mock(EngineAddressResolver.class);
            GrpcChannelFactory channelFactory = mock(GrpcChannelFactory.class);
            GrpcReporter grpcReporter = mock(GrpcReporter.class);
            when(channelFactory.create(new GrpcTarget("127.0.0.1", server.getPort()))).thenReturn(channel);
            client = new EngineGrpcClient(
                    addressResolver, channelFactory, grpcReporter, mock(GrpcRuntimeMetrics.class));

            EngineRpcService.KvCacheGroupListPB response = client.getKvCacheGroupsMetadata(
                    "127.0.0.1",
                    server.getPort(),
                    EngineRpcService.KvCacheGroupsRequestPB.getDefaultInstance(),
                    1_000)
                    .block();

            assertEquals(1, response.getItemsCount());
        } finally {
            close(client, server, channel);
        }
    }

    @Test
    void returnsMultimodalWorkerStatusSynchronously() throws Exception {
        Server server = ServerBuilder.forPort(0)
                .addService(new MultimodalRpcServiceGrpc.MultimodalRpcServiceImplBase() {
                    @Override
                    public void getWorkerStatus(
                            EngineRpcService.StatusVersionPB request,
                            StreamObserver<EngineRpcService.WorkerStatusPB> responseObserver) {
                        responseObserver.onNext(EngineRpcService.WorkerStatusPB.newBuilder()
                                .setStatusVersion(88)
                                .build());
                        responseObserver.onCompleted();
                    }
                })
                .build()
                .start();
        ManagedChannel channel = ManagedChannelBuilder.forAddress("127.0.0.1", server.getPort())
                .usePlaintext()
                .build();
        EngineGrpcClient client = null;
        try {
            EngineAddressResolver addressResolver = mock(EngineAddressResolver.class);
            GrpcChannelFactory channelFactory = mock(GrpcChannelFactory.class);
            GrpcReporter grpcReporter = mock(GrpcReporter.class);
            when(channelFactory.create(new GrpcTarget("127.0.0.1", server.getPort()))).thenReturn(channel);
            client = new EngineGrpcClient(
                    addressResolver, channelFactory, grpcReporter, mock(GrpcRuntimeMetrics.class));

            EngineRpcService.WorkerStatusPB response = client.getMultimodalWorkerStatus(
                    "127.0.0.1",
                    server.getPort(),
                    EngineRpcService.StatusVersionPB.getDefaultInstance(),
                    1_000);

            org.junit.jupiter.api.Assertions.assertEquals(88, response.getStatusVersion());
        } finally {
            close(client, server, channel);
        }
    }

    @Test
    void returnsMultimodalCacheStatusSynchronously() throws Exception {
        Server server = ServerBuilder.forPort(0)
                .addService(new MultimodalRpcServiceGrpc.MultimodalRpcServiceImplBase() {
                    @Override
                    public void getCacheStatus(
                            EngineRpcService.CacheVersionPB request,
                            StreamObserver<EngineRpcService.CacheStatusPB> responseObserver) {
                        responseObserver.onNext(EngineRpcService.CacheStatusPB.newBuilder()
                                .setVersion(44)
                                .build());
                        responseObserver.onCompleted();
                    }
                })
                .build()
                .start();
        ManagedChannel channel = ManagedChannelBuilder.forAddress("127.0.0.1", server.getPort())
                .usePlaintext()
                .build();
        EngineGrpcClient client = null;
        try {
            EngineAddressResolver addressResolver = mock(EngineAddressResolver.class);
            GrpcChannelFactory channelFactory = mock(GrpcChannelFactory.class);
            GrpcReporter grpcReporter = mock(GrpcReporter.class);
            when(channelFactory.create(new GrpcTarget("127.0.0.1", server.getPort()))).thenReturn(channel);
            client = new EngineGrpcClient(
                    addressResolver, channelFactory, grpcReporter, mock(GrpcRuntimeMetrics.class));

            EngineRpcService.CacheStatusPB response = client.getMultimodalCacheStatus(
                    "127.0.0.1",
                    server.getPort(),
                    EngineRpcService.CacheVersionPB.getDefaultInstance(),
                    1_000);

            assertEquals(44, response.getVersion());
        } finally {
            close(client, server, channel);
        }
    }

    @Test
    void propagatesGrpcErrorsSynchronously() throws Exception {
        Server server = ServerBuilder.forPort(0)
                .addService(new RpcServiceGrpc.RpcServiceImplBase() {
                    @Override
                    public void getWorkerStatus(
                            EngineRpcService.StatusVersionPB request,
                            StreamObserver<EngineRpcService.WorkerStatusPB> responseObserver) {
                        responseObserver.onError(Status.INVALID_ARGUMENT
                                .withDescription("invalid request")
                                .asRuntimeException());
                    }
                })
                .build()
                .start();
        ManagedChannel channel = ManagedChannelBuilder.forAddress("127.0.0.1", server.getPort())
                .usePlaintext()
                .build();
        EngineGrpcClient client = null;
        try {
            EngineAddressResolver addressResolver = mock(EngineAddressResolver.class);
            GrpcChannelFactory channelFactory = mock(GrpcChannelFactory.class);
            GrpcReporter grpcReporter = mock(GrpcReporter.class);
            when(channelFactory.create(new GrpcTarget("127.0.0.1", server.getPort()))).thenReturn(channel);
            client = new EngineGrpcClient(
                    addressResolver, channelFactory, grpcReporter, mock(GrpcRuntimeMetrics.class));
            EngineGrpcClient grpcClient = client;

            StatusRuntimeException exception = assertThrows(StatusRuntimeException.class, () -> grpcClient.getWorkerStatus(
                    "127.0.0.1",
                    server.getPort(),
                    EngineRpcService.StatusVersionPB.getDefaultInstance(),
                    1_000));

            assertEquals(Status.Code.INVALID_ARGUMENT, exception.getStatus().getCode());
        } finally {
            close(client, server, channel);
        }
    }

    @Test
    void propagatesGrpcDeadlineSynchronously() throws Exception {
        Server server = ServerBuilder.forPort(0)
                .addService(new RpcServiceGrpc.RpcServiceImplBase() {
                    @Override
                    public void getWorkerStatus(
                            EngineRpcService.StatusVersionPB request,
                            StreamObserver<EngineRpcService.WorkerStatusPB> responseObserver) {
                    }
                })
                .build()
                .start();
        ManagedChannel channel = ManagedChannelBuilder.forAddress("127.0.0.1", server.getPort())
                .usePlaintext()
                .build();
        EngineGrpcClient client = null;
        try {
            EngineAddressResolver addressResolver = mock(EngineAddressResolver.class);
            GrpcChannelFactory channelFactory = mock(GrpcChannelFactory.class);
            GrpcReporter grpcReporter = mock(GrpcReporter.class);
            when(channelFactory.create(new GrpcTarget("127.0.0.1", server.getPort()))).thenReturn(channel);
            client = new EngineGrpcClient(
                    addressResolver, channelFactory, grpcReporter, mock(GrpcRuntimeMetrics.class));
            EngineGrpcClient grpcClient = client;

            StatusRuntimeException exception = assertThrows(StatusRuntimeException.class, () -> grpcClient.getWorkerStatus(
                    "127.0.0.1",
                    server.getPort(),
                    EngineRpcService.StatusVersionPB.getDefaultInstance(),
                    100));

            assertEquals(Status.Code.DEADLINE_EXCEEDED, exception.getStatus().getCode());
        } finally {
            close(client, server, channel);
        }
    }

    @Test
    void retriesBrokenConnectionOnceAndReportsRetryMetrics() throws Exception {
        AtomicInteger requestCount = new AtomicInteger();
        Server server = ServerBuilder.forPort(0)
                .addService(new RpcServiceGrpc.RpcServiceImplBase() {
                    @Override
                    public void getWorkerStatus(
                            EngineRpcService.StatusVersionPB request,
                            StreamObserver<EngineRpcService.WorkerStatusPB> responseObserver) {
                        if (requestCount.getAndIncrement() == 0) {
                            responseObserver.onError(Status.UNAVAILABLE
                                    .withDescription("Broken pipe")
                                    .asRuntimeException());
                            return;
                        }
                        responseObserver.onNext(EngineRpcService.WorkerStatusPB.newBuilder()
                                .setStatusVersion(99)
                                .build());
                        responseObserver.onCompleted();
                    }
                })
                .build()
                .start();
        ManagedChannel firstChannel = ManagedChannelBuilder.forAddress("127.0.0.1", server.getPort())
                .usePlaintext()
                .build();
        ManagedChannel retryChannel = ManagedChannelBuilder.forAddress("127.0.0.1", server.getPort())
                .usePlaintext()
                .build();
        EngineGrpcClient client = null;
        try {
            EngineAddressResolver addressResolver = mock(EngineAddressResolver.class);
            GrpcChannelFactory channelFactory = mock(GrpcChannelFactory.class);
            GrpcReporter grpcReporter = mock(GrpcReporter.class);
            GrpcTarget target = new GrpcTarget("127.0.0.1", server.getPort());
            when(channelFactory.create(target)).thenReturn(firstChannel, retryChannel);
            client = new EngineGrpcClient(
                    addressResolver, channelFactory, grpcReporter, mock(GrpcRuntimeMetrics.class));

            EngineRpcService.WorkerStatusPB response = client.getWorkerStatus(
                    "127.0.0.1",
                    server.getPort(),
                    EngineRpcService.StatusVersionPB.getDefaultInstance(),
                    1_000);

            assertEquals(99, response.getStatusVersion());
            assertEquals(2, requestCount.get());
            verify(channelFactory, times(2)).create(target);
            verify(grpcReporter).reportConnectionDuration(eq("127.0.0.1"), eq("GetWorkerStatus"), anyLong());
            verify(grpcReporter).reportCallMetrics(
                    eq("127.0.0.1"), eq("GetWorkerStatus"), anyLong(), anyInt(), eq(true));
        } finally {
            close(client, server, firstChannel, retryChannel);
        }
    }

    @Test
    void failsAfterTheSingleBrokenConnectionRetry() throws Exception {
        AtomicInteger requestCount = new AtomicInteger();
        Server server = ServerBuilder.forPort(0)
                .addService(new RpcServiceGrpc.RpcServiceImplBase() {
                    @Override
                    public void getWorkerStatus(
                            EngineRpcService.StatusVersionPB request,
                            StreamObserver<EngineRpcService.WorkerStatusPB> responseObserver) {
                        requestCount.incrementAndGet();
                        responseObserver.onError(Status.UNAVAILABLE
                                .withDescription("Broken pipe")
                                .asRuntimeException());
                    }
                })
                .build()
                .start();
        ManagedChannel firstChannel = ManagedChannelBuilder.forAddress("127.0.0.1", server.getPort())
                .usePlaintext()
                .build();
        ManagedChannel retryChannel = ManagedChannelBuilder.forAddress("127.0.0.1", server.getPort())
                .usePlaintext()
                .build();
        EngineGrpcClient client = null;
        try {
            EngineAddressResolver addressResolver = mock(EngineAddressResolver.class);
            GrpcChannelFactory channelFactory = mock(GrpcChannelFactory.class);
            GrpcReporter grpcReporter = mock(GrpcReporter.class);
            GrpcTarget target = new GrpcTarget("127.0.0.1", server.getPort());
            when(channelFactory.create(target)).thenReturn(firstChannel, retryChannel);
            client = new EngineGrpcClient(
                    addressResolver, channelFactory, grpcReporter, mock(GrpcRuntimeMetrics.class));
            EngineGrpcClient grpcClient = client;

            StatusRuntimeException exception = assertThrows(StatusRuntimeException.class,
                    () -> grpcClient.getWorkerStatus(
                            "127.0.0.1",
                            server.getPort(),
                            EngineRpcService.StatusVersionPB.getDefaultInstance(),
                            1_000));

            assertEquals(Status.Code.UNAVAILABLE, exception.getStatus().getCode());
            assertEquals(2, requestCount.get());
            verify(channelFactory, times(2)).create(target);
        } finally {
            close(client, server, firstChannel, retryChannel);
        }
    }

    @Test
    void reusesChannelsAndClosesThemWhenWorkerGoesOffline() {
        EngineAddressResolver addressResolver = mock(EngineAddressResolver.class);
        GrpcChannelFactory channelFactory = mock(GrpcChannelFactory.class);
        GrpcReporter grpcReporter = mock(GrpcReporter.class);
        List<ManagedChannel> createdChannels = new ArrayList<>();
        when(channelFactory.create(new GrpcTarget("10.0.0.1", 8081)))
                .thenAnswer(invocation -> {
                    ManagedChannel channel = mock(ManagedChannel.class);
                    createdChannels.add(channel);
                    return channel;
                });

        EngineGrpcClient client =
                new EngineGrpcClient(addressResolver, channelFactory, grpcReporter, mock(GrpcRuntimeMetrics.class));
        ArgumentCaptor<EngineAddressResolver.Listener> listener =
                ArgumentCaptor.forClass(EngineAddressResolver.Listener.class);
        verify(addressResolver).subscribe(listener.capture());

        listener.getValue().onAddressUpdate(List.of("10.0.0.1:8080"));
        listener.getValue().onAddressUpdate(List.of("10.0.0.1:8080"));

        verify(channelFactory, times(5)).create(new GrpcTarget("10.0.0.1", 8081));

        listener.getValue().onAddressUpdate(List.of());
        for (ManagedChannel channel : createdChannels) {
            verify(channel).shutdown();
        }

        client.shutdown();
    }

    private static void close(EngineGrpcClient client, Server server, ManagedChannel... channels) throws InterruptedException {
        if (client != null) {
            client.shutdown();
        } else {
            for (ManagedChannel channel : channels) {
                channel.shutdownNow();
            }
        }
        for (ManagedChannel channel : channels) {
            channel.awaitTermination(1, TimeUnit.SECONDS);
        }
        server.shutdownNow();
        server.awaitTermination(1, TimeUnit.SECONDS);
    }
}
