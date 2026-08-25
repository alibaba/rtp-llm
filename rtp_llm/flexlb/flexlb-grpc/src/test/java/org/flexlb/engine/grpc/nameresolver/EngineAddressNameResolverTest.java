package org.flexlb.engine.grpc.nameresolver;

import org.junit.jupiter.api.Test;
import org.mockito.ArgumentCaptor;

import java.util.List;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verify;

class EngineAddressNameResolverTest {

    @Test
    void forwardsAddressUpdatesFromSharedResolverToLegacyListener() {
        EngineAddressResolver engineAddressResolver = mock(EngineAddressResolver.class);
        CustomNameResolver.Listener legacyListener = mock(CustomNameResolver.Listener.class);
        EngineAddressNameResolver adapter =
                new EngineAddressNameResolver(engineAddressResolver);

        adapter.start(legacyListener);

        ArgumentCaptor<EngineAddressResolver.Listener> listenerCaptor =
                ArgumentCaptor.forClass(EngineAddressResolver.Listener.class);
        verify(engineAddressResolver).subscribe(listenerCaptor.capture());

        List<String> addresses = List.of("10.0.0.1:8080", "10.0.0.2:8080");
        listenerCaptor.getValue().onAddressUpdate(addresses);

        verify(legacyListener).onAddressUpdate(addresses);
    }
}
