package org.flexlb.engine.grpc.client;

import io.grpc.ManagedChannel;
import org.flexlb.engine.grpc.core.GrpcChannelFactory;
import org.flexlb.engine.grpc.core.GrpcTarget;
import org.flexlb.engine.grpc.monitor.GrpcReporter;
import org.flexlb.engine.grpc.nameresolver.EngineAddressResolver;
import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;

import java.util.ArrayList;
import java.util.List;

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
        List<ManagedChannel> createdChannels = new ArrayList<>();
        when(channelFactory.create(new GrpcTarget("10.0.0.1", 8081)))
                .thenAnswer(invocation -> {
                    ManagedChannel channel = mock(ManagedChannel.class);
                    createdChannels.add(channel);
                    return channel;
                });

        EngineGrpcClient client =
                new EngineGrpcClient(addressResolver, channelFactory, grpcReporter);
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
}
