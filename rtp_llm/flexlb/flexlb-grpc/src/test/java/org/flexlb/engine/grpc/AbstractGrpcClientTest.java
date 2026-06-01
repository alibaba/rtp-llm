package org.flexlb.engine.grpc;

import io.grpc.ManagedChannel;
import org.flexlb.cache.core.EngineLocalView;
import org.flexlb.cache.core.GlobalCacheIndex;
import org.flexlb.engine.grpc.monitor.GrpcReporter;
import org.junit.jupiter.api.Test;

import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;

class AbstractGrpcClientTest {

    @Test
    void address_update_manages_all_rpc_service_types() {
        TestGrpcClient client = new TestGrpcClient();

        client.onAddressUpdate(List.of("10.0.0.1:8080"));

        for (AbstractGrpcClient.ServiceType serviceType : AbstractGrpcClient.ServiceType.values()) {
            assertTrue(client.hasChannel("10.0.0.1", 8081, serviceType));
        }
        assertEquals(AbstractGrpcClient.ServiceType.values().length, client.channelCount());

        client.onAddressUpdate(List.of("10.0.0.1:8080"));

        assertEquals(AbstractGrpcClient.ServiceType.values().length, client.channelCount());
    }

    private static final class TestGrpcClient extends AbstractGrpcClient<AbstractGrpcClient.GrpcStubWrapper> {

        private TestGrpcClient() {
            super(mock(EngineLocalView.class), mock(GlobalCacheIndex.class), mock(GrpcReporter.class));
        }

        @Override
        protected ManagedChannel createChannel(String channelKey) {
            return mock(ManagedChannel.class);
        }

        @Override
        protected AbstractGrpcClient.GrpcStubWrapper createStub(ManagedChannel channel) {
            return new AbstractGrpcClient.GrpcStubWrapper(null, null);
        }

        private boolean hasChannel(String ip, int port, ServiceType serviceType) {
            return channelPool.containsKey(createKey(ip, port, serviceType));
        }

        private int channelCount() {
            return channelPool.size();
        }
    }
}
