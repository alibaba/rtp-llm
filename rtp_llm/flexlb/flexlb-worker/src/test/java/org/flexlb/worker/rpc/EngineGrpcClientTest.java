package org.flexlb.worker.rpc;

import io.grpc.ManagedChannel;
import io.grpc.Server;
import io.grpc.inprocess.InProcessChannelBuilder;
import io.grpc.inprocess.InProcessServerBuilder;
import io.grpc.stub.StreamObserver;
import org.flexlb.config.cache.CacheMatchConfiguration;
import org.flexlb.util.discovery.DiscoveredHost;
import org.flexlb.util.metrics.GrpcReporter;
import org.flexlb.util.network.GrpcChannelFactory;
import org.flexlb.util.network.GrpcTarget;
import org.flexlb.worker.rpc.protocol.EngineRpcService;
import org.flexlb.worker.rpc.protocol.RpcServiceGrpc;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Checks engine RPC request construction and client behavior.
 */
class EngineGrpcClientTest {

    @Test
    void reusesChannelsAndClosesThemWhenWorkerGoesOffline() {
        EngineAddressResolver addressResolver = mock(EngineAddressResolver.class);
        GrpcChannelFactory channelFactory = mock(GrpcChannelFactory.class);
        GrpcReporter grpcReporter = mock(GrpcReporter.class);
        CacheMatchConfiguration cacheMatchConfiguration = mock(CacheMatchConfiguration.class);
        List<ManagedChannel> createdChannels = new ArrayList<>();
        when(channelFactory.create(any()))
                .thenAnswer(
                        invocation -> {
                            ManagedChannel channel = mock(ManagedChannel.class);
                            createdChannels.add(channel);
                            return channel;
                        });

        EngineGrpcClient client =
                createClient(
                        addressResolver, channelFactory, grpcReporter, cacheMatchConfiguration);
        ArgumentCaptor<EngineAddressResolver.Listener> listener =
                ArgumentCaptor.forClass(EngineAddressResolver.Listener.class);
        verify(addressResolver).subscribe(listener.capture());

        DiscoveredHost host = new DiscoveredHost("10.0.0.1", 8080, 8081, 8085, 18002, "", "", "");
        listener.getValue().onAddressUpdate(List.of(host));
        listener.getValue().onAddressUpdate(List.of(host));

        verify(channelFactory, times(1)).create(new GrpcTarget("10.0.0.1", 8081));
        verify(channelFactory, times(4)).create(new GrpcTarget("10.0.0.1", 18002));

        listener.getValue().onAddressUpdate(List.of());
        for (ManagedChannel channel : createdChannels) {
            verify(channel).shutdown();
        }

        client.shutdownChannelPool();
    }

    @Test
    void doesNotCreateCacheStatusChannelsWhenKvcmIsEnabled() {
        EngineAddressResolver addressResolver = mock(EngineAddressResolver.class);
        GrpcChannelFactory channelFactory = mock(GrpcChannelFactory.class);
        GrpcReporter grpcReporter = mock(GrpcReporter.class);
        CacheMatchConfiguration cacheMatchConfiguration = mock(CacheMatchConfiguration.class);
        when(cacheMatchConfiguration.isKvcmEnabled()).thenReturn(true);
        when(channelFactory.create(any())).thenReturn(mock(ManagedChannel.class));

        EngineGrpcClient client =
                createClient(
                        addressResolver, channelFactory, grpcReporter, cacheMatchConfiguration);
        ArgumentCaptor<EngineAddressResolver.Listener> listener =
                ArgumentCaptor.forClass(EngineAddressResolver.Listener.class);
        verify(addressResolver).subscribe(listener.capture());

        listener.getValue()
                .onAddressUpdate(
                        List.of(
                                new DiscoveredHost(
                                        "10.0.0.1", 8080, 8081, 8085, 18002, "", "", "")));

        verify(channelFactory, times(1)).create(new GrpcTarget("10.0.0.1", 8081));
        verify(channelFactory, times(2)).create(new GrpcTarget("10.0.0.1", 18002));

        client.shutdownChannelPool();
    }

    @Test
    void sharesDataPlaneChannelsAcrossLogicalEnginesAndKeepsStatusPortsSeparate() {
        EngineAddressResolver addressResolver = mock(EngineAddressResolver.class);
        GrpcChannelFactory channelFactory = mock(GrpcChannelFactory.class);
        GrpcReporter grpcReporter = mock(GrpcReporter.class);
        CacheMatchConfiguration cacheMatchConfiguration = mock(CacheMatchConfiguration.class);
        when(channelFactory.create(any())).thenReturn(mock(ManagedChannel.class));
        EngineGrpcClient client =
                createClient(
                        addressResolver, channelFactory, grpcReporter, cacheMatchConfiguration);
        ArgumentCaptor<EngineAddressResolver.Listener> listener =
                ArgumentCaptor.forClass(EngineAddressResolver.Listener.class);
        verify(addressResolver).subscribe(listener.capture());

        listener.getValue()
                .onAddressUpdate(
                        List.of(
                                new DiscoveredHost(
                                        "10.0.0.1", 8080, 8081, 8085, 18002, "", "", "", 0, 2),
                                new DiscoveredHost(
                                        "10.0.0.1", 8080, 8081, 8085, 18003, "", "", "", 1, 2)));

        verify(channelFactory, times(1)).create(new GrpcTarget("10.0.0.1", 8081));
        verify(channelFactory, times(4)).create(new GrpcTarget("10.0.0.1", 18002));
        verify(channelFactory, times(4)).create(new GrpcTarget("10.0.0.1", 18003));
        client.shutdownChannelPool();
    }

    @Test
    void workerStatusAsyncReusesPrecreatedStatusChannel() throws Exception {
        String serverName = InProcessServerBuilder.generateName();
        Server server =
                InProcessServerBuilder.forName(serverName)
                        .directExecutor()
                        .addService(
                                new RpcServiceGrpc.RpcServiceImplBase() {
                                    @Override
                                    public void getWorkerStatus(
                                            EngineRpcService.StatusVersionPB request,
                                            StreamObserver<EngineRpcService.WorkerStatusPB>
                                                    observer) {
                                        observer.onNext(
                                                EngineRpcService.WorkerStatusPB.newBuilder()
                                                        .setAlive(true)
                                                        .build());
                                        observer.onCompleted();
                                    }
                                })
                        .build()
                        .start();
        EngineAddressResolver addressResolver = mock(EngineAddressResolver.class);
        GrpcChannelFactory channelFactory = mock(GrpcChannelFactory.class);
        GrpcReporter grpcReporter = mock(GrpcReporter.class);
        CacheMatchConfiguration cacheMatchConfiguration = mock(CacheMatchConfiguration.class);
        when(channelFactory.create(any()))
                .thenAnswer(
                        invocation ->
                                InProcessChannelBuilder.forName(serverName)
                                        .propagateCauseWithStatus(true)
                                        .directExecutor()
                                        .build());
        EngineGrpcClient client =
                createClient(
                        addressResolver, channelFactory, grpcReporter, cacheMatchConfiguration);
        ArgumentCaptor<EngineAddressResolver.Listener> listener =
                ArgumentCaptor.forClass(EngineAddressResolver.Listener.class);
        verify(addressResolver).subscribe(listener.capture());

        try {
            listener.getValue()
                    .onAddressUpdate(
                            List.of(
                                    new DiscoveredHost(
                                            "10.0.0.1", 8080, 8081, 8085, 18002, "", "", "")));

            assertTrue(
                    client.getWorkerStatusAsync(
                                    "10.0.0.1",
                                    18002,
                                    EngineRpcService.StatusVersionPB.getDefaultInstance(),
                                    1000)
                            .get(2, TimeUnit.SECONDS)
                            .getAlive());
            verify(channelFactory, times(4)).create(new GrpcTarget("10.0.0.1", 18002));
        } finally {
            client.shutdownChannelPool();
            server.shutdownNow().awaitTermination(2, TimeUnit.SECONDS);
        }
    }

    private static EngineGrpcClient createClient(
            EngineAddressResolver resolver,
            GrpcChannelFactory factory,
            GrpcReporter reporter,
            CacheMatchConfiguration cacheConfig) {
        return new EngineGrpcClient(
                resolver,
                mock(java.util.concurrent.ThreadPoolExecutor.class),
                mock(io.netty.channel.EventLoopGroup.class),
                reporter,
                2000,
                cacheConfig) {
            @Override
            protected ManagedChannel createChannel(String key) {
                int serviceSeparator = key.lastIndexOf(':');
                int portSeparator = key.lastIndexOf(':', serviceSeparator - 1);
                return factory.create(
                        new GrpcTarget(
                                key.substring(0, portSeparator),
                                Integer.parseInt(
                                        key.substring(portSeparator + 1, serviceSeparator))));
            }
        };
    }
}
