package org.flexlb.engine.grpc.client;

import io.grpc.ManagedChannel;
import io.grpc.Server;
import io.grpc.inprocess.InProcessChannelBuilder;
import io.grpc.inprocess.InProcessServerBuilder;
import io.grpc.stub.StreamObserver;
import org.flexlb.config.CacheMatchConfiguration;
import org.flexlb.dao.master.WorkerHost;
import org.flexlb.engine.grpc.EngineRpcService;
import org.flexlb.engine.grpc.RpcServiceGrpc;
import org.flexlb.engine.grpc.core.GrpcChannelFactory;
import org.flexlb.engine.grpc.core.GrpcTarget;
import org.flexlb.engine.grpc.monitor.GrpcReporter;
import org.flexlb.engine.grpc.nameresolver.EngineAddressResolver;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyInt;
import static org.mockito.ArgumentMatchers.anyLong;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

class EngineGrpcClientTest {

    @Test
    void reusesChannelsAndClosesThemWhenWorkerGoesOffline() {
        EngineAddressResolver addressResolver = mock(EngineAddressResolver.class);
        GrpcChannelFactory channelFactory = mock(GrpcChannelFactory.class);
        GrpcReporter grpcReporter = mock(GrpcReporter.class);
        CacheMatchConfiguration cacheMatchConfiguration = mock(CacheMatchConfiguration.class);
        List<ManagedChannel> createdChannels = new ArrayList<>();
        when(channelFactory.create(any()))
                .thenAnswer(invocation -> {
                    ManagedChannel channel = mock(ManagedChannel.class);
                    createdChannels.add(channel);
                    return channel;
                });

        EngineGrpcClient client =
                new EngineGrpcClient(
                        addressResolver, channelFactory, grpcReporter, cacheMatchConfiguration);
        ArgumentCaptor<EngineAddressResolver.Listener> listener =
                ArgumentCaptor.forClass(EngineAddressResolver.Listener.class);
        verify(addressResolver).subscribe(listener.capture());

        WorkerHost host = new WorkerHost(
                "10.0.0.1", 8080, 8081, 8085, 18002, "", "", "");
        listener.getValue().onAddressUpdate(List.of(host));
        listener.getValue().onAddressUpdate(List.of(host));

        verify(channelFactory, times(2)).create(new GrpcTarget("10.0.0.1", 8081));
        verify(channelFactory, times(4)).create(new GrpcTarget("10.0.0.1", 18002));

        listener.getValue().onAddressUpdate(List.of());
        for (ManagedChannel channel : createdChannels) {
            verify(channel).shutdown();
        }

        client.shutdown();
    }

    @Test
    void doesNotCreateCacheStatusChannelsWhenKvcmIsEnabled() {
        EngineAddressResolver addressResolver = mock(EngineAddressResolver.class);
        GrpcChannelFactory channelFactory = mock(GrpcChannelFactory.class);
        GrpcReporter grpcReporter = mock(GrpcReporter.class);
        CacheMatchConfiguration cacheMatchConfiguration = mock(CacheMatchConfiguration.class);
        when(cacheMatchConfiguration.isKvcmEnabled()).thenReturn(true);
        when(channelFactory.create(any())).thenReturn(mock(ManagedChannel.class));

        EngineGrpcClient client = new EngineGrpcClient(
                addressResolver, channelFactory, grpcReporter, cacheMatchConfiguration);
        ArgumentCaptor<EngineAddressResolver.Listener> listener =
                ArgumentCaptor.forClass(EngineAddressResolver.Listener.class);
        verify(addressResolver).subscribe(listener.capture());

        listener.getValue().onAddressUpdate(List.of(new WorkerHost(
                "10.0.0.1", 8080, 8081, 8085, 18002, "", "", "")));

        verify(channelFactory, times(2)).create(new GrpcTarget("10.0.0.1", 8081));
        verify(channelFactory, times(2)).create(new GrpcTarget("10.0.0.1", 18002));

        client.shutdown();
    }

    @Test
    void sharesDataPlaneChannelsAcrossLogicalEnginesAndKeepsStatusPortsSeparate() {
        EngineAddressResolver addressResolver = mock(EngineAddressResolver.class);
        GrpcChannelFactory channelFactory = mock(GrpcChannelFactory.class);
        GrpcReporter grpcReporter = mock(GrpcReporter.class);
        CacheMatchConfiguration cacheMatchConfiguration = mock(CacheMatchConfiguration.class);
        when(channelFactory.create(any())).thenReturn(mock(ManagedChannel.class));
        EngineGrpcClient client = new EngineGrpcClient(
                addressResolver, channelFactory, grpcReporter, cacheMatchConfiguration);
        ArgumentCaptor<EngineAddressResolver.Listener> listener =
                ArgumentCaptor.forClass(EngineAddressResolver.Listener.class);
        verify(addressResolver).subscribe(listener.capture());

        listener.getValue().onAddressUpdate(List.of(
                new WorkerHost(
                        "10.0.0.1", 8080, 8081, 8085, 18002,
                        "", "", "", 0, 2),
                new WorkerHost(
                        "10.0.0.1", 8080, 8081, 8085, 18003,
                        "", "", "", 1, 2)));

        verify(channelFactory, times(2)).create(new GrpcTarget("10.0.0.1", 8081));
        verify(channelFactory, times(4)).create(new GrpcTarget("10.0.0.1", 18002));
        verify(channelFactory, times(4)).create(new GrpcTarget("10.0.0.1", 18003));
        client.shutdown();
    }

    @Test
    void workerStatusAsyncReusesPrecreatedStatusChannel() throws Exception {
        String serverName = InProcessServerBuilder.generateName();
        Server server = InProcessServerBuilder.forName(serverName).directExecutor()
                .addService(new RpcServiceGrpc.RpcServiceImplBase() {
                    @Override
                    public void getWorkerStatus(
                            EngineRpcService.StatusVersionPB request,
                            StreamObserver<EngineRpcService.WorkerStatusPB> observer) {
                        observer.onNext(EngineRpcService.WorkerStatusPB.newBuilder()
                                .setAlive(true)
                                .build());
                        observer.onCompleted();
                    }
                }).build().start();
        EngineAddressResolver addressResolver = mock(EngineAddressResolver.class);
        GrpcChannelFactory channelFactory = mock(GrpcChannelFactory.class);
        GrpcReporter grpcReporter = mock(GrpcReporter.class);
        CacheMatchConfiguration cacheMatchConfiguration = mock(CacheMatchConfiguration.class);
        when(channelFactory.create(any())).thenAnswer(invocation ->
                InProcessChannelBuilder.forName(serverName).directExecutor().build());
        EngineGrpcClient client = new EngineGrpcClient(
                addressResolver, channelFactory, grpcReporter, cacheMatchConfiguration);
        ArgumentCaptor<EngineAddressResolver.Listener> listener =
                ArgumentCaptor.forClass(EngineAddressResolver.Listener.class);
        verify(addressResolver).subscribe(listener.capture());

        try {
            listener.getValue().onAddressUpdate(List.of(new WorkerHost(
                    "10.0.0.1", 8080, 8081, 8085, 18002, "", "", "")));

            assertTrue(client.getWorkerStatusAsync(
                            "10.0.0.1",
                            18002,
                            EngineRpcService.StatusVersionPB.getDefaultInstance(),
                            1000)
                    .get(2, TimeUnit.SECONDS)
                    .getAlive());
            verify(channelFactory, times(4)).create(new GrpcTarget("10.0.0.1", 18002));
        } finally {
            client.shutdown();
            server.shutdownNow().awaitTermination(2, TimeUnit.SECONDS);
        }
    }

    @Test
    void retriesStatusButDoesNotRetryBatchAfterBrokenConnection() throws Exception {
        String serverName = InProcessServerBuilder.generateName();
        AtomicInteger workerStatusCalls = new AtomicInteger();
        AtomicInteger batchCalls = new AtomicInteger();
        AtomicInteger cancelCalls = new AtomicInteger();
        Server server = InProcessServerBuilder.forName(serverName).directExecutor()
                .addService(new RpcServiceGrpc.RpcServiceImplBase() {
                    @Override
                    public void getWorkerStatus(
                            EngineRpcService.StatusVersionPB request,
                            StreamObserver<EngineRpcService.WorkerStatusPB> observer) {
                        if (workerStatusCalls.incrementAndGet() == 1) {
                            observer.onError(io.grpc.Status.INTERNAL
                                    .withDescription("Connection reset")
                                    .asRuntimeException());
                            return;
                        }
                        observer.onNext(EngineRpcService.WorkerStatusPB.newBuilder()
                                .setAlive(true)
                                .build());
                        observer.onCompleted();
                    }

                    @Override
                    public void enqueueBatch(
                            EngineRpcService.EnqueueBatchRequestPB request,
                            StreamObserver<EngineRpcService.EnqueueBatchResponsePB> observer) {
                        batchCalls.incrementAndGet();
                        observer.onError(io.grpc.Status.INTERNAL
                                .withDescription("Connection reset")
                                .asRuntimeException());
                    }

                    @Override
                    public void cancel(
                            EngineRpcService.CancelRequestPB request,
                            StreamObserver<EngineRpcService.CancelResponsePB> observer) {
                        cancelCalls.incrementAndGet();
                        observer.onError(io.grpc.Status.INTERNAL
                                .withDescription("Connection reset")
                                .asRuntimeException());
                    }
                }).build().start();
        EngineAddressResolver addressResolver = mock(EngineAddressResolver.class);
        GrpcChannelFactory channelFactory = mock(GrpcChannelFactory.class);
        GrpcReporter grpcReporter = mock(GrpcReporter.class);
        when(channelFactory.create(any())).thenAnswer(invocation ->
                InProcessChannelBuilder.forName(serverName).directExecutor().build());
        EngineGrpcClient client = new EngineGrpcClient(
                addressResolver,
                channelFactory,
                grpcReporter,
                mock(CacheMatchConfiguration.class));
        ArgumentCaptor<EngineAddressResolver.Listener> listener =
                ArgumentCaptor.forClass(EngineAddressResolver.Listener.class);
        verify(addressResolver).subscribe(listener.capture());

        try {
            listener.getValue().onAddressUpdate(List.of(new WorkerHost(
                    "10.0.0.1", 8080, 8081, 8085, 18002, "", "", "")));

            assertTrue(client.getWorkerStatusAsync(
                            "10.0.0.1",
                            18002,
                            EngineRpcService.StatusVersionPB.getDefaultInstance(),
                            1000)
                    .get(2, TimeUnit.SECONDS)
                    .getAlive());
            assertThrows(Exception.class, () -> client.batchEnqueueAsync(
                            "10.0.0.1",
                            8081,
                            EngineRpcService.EnqueueBatchRequestPB.getDefaultInstance(),
                            1000)
                    .get(2, TimeUnit.SECONDS));
            assertThrows(Exception.class, () -> client.cancelAsync(
                            "10.0.0.1",
                            8081,
                            EngineRpcService.CancelRequestPB.getDefaultInstance(),
                            1000)
                    .get(2, TimeUnit.SECONDS));

            assertEquals(2, workerStatusCalls.get());
            assertEquals(1, batchCalls.get());
            assertEquals(1, cancelCalls.get());
            verify(grpcReporter).reportConnectionDuration(eq("GetWorkerStatus"), anyLong());
            verify(grpcReporter).reportCallMetrics(
                    eq("GetWorkerStatus"), anyLong(), anyInt(), eq(true));
            verify(channelFactory, times(5)).create(new GrpcTarget("10.0.0.1", 18002));
            verify(channelFactory, times(2)).create(new GrpcTarget("10.0.0.1", 8081));
        } finally {
            client.shutdown();
            server.shutdownNow().awaitTermination(2, TimeUnit.SECONDS);
        }
    }
}
