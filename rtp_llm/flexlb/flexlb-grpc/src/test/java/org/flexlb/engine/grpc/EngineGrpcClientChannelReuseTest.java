package org.flexlb.engine.grpc;

import io.grpc.ManagedChannel;
import io.grpc.Server;
import io.grpc.inprocess.InProcessChannelBuilder;
import io.grpc.inprocess.InProcessServerBuilder;
import io.grpc.stub.StreamObserver;
import io.netty.channel.EventLoopGroup;
import org.flexlb.engine.grpc.monitor.GrpcReporter;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.mockito.Mockito.mock;

class EngineGrpcClientChannelReuseTest {

    @ParameterizedTest
    @CsvSource({"30670, 18002, 1", "30670, 18002, 2", "8001, 18002, 1", "8001, 18002, 2", "8001, 8002, 1"})
    void statusRpcReusesPortAcrossDiscoveryRefreshAndClosesOfflineChannels(int httpPort, int statusPort, int engineCount) throws Exception {
        String serverName = InProcessServerBuilder.generateName();
        Server server = InProcessServerBuilder.forName(serverName).directExecutor()
                .addService(new RpcServiceGrpc.RpcServiceImplBase() {
                    @Override
                    public void getWorkerStatus(EngineRpcService.StatusVersionPB request,
                                                StreamObserver<EngineRpcService.WorkerStatusPB> observer) {
                        observer.onNext(EngineRpcService.WorkerStatusPB.newBuilder()
                                .setAlive(true).setRole("PREFILL").build());
                        observer.onCompleted();
                    }
                }).build().start();
        ThreadPoolExecutor executor = (ThreadPoolExecutor) Executors.newFixedThreadPool(1);
        RecordingClient client = new RecordingClient(serverName, executor);
        try {
            List<String> hosts = Collections.nCopies(engineCount, "10.0.0.1:" + httpPort);
            client.onAddressUpdate(hosts);
            assertEquals(6, client.created.size());
            for (int index = 0; index < engineCount; index++) {
                assertTrue(client.getWorkerStatusAsync("10.0.0.1", statusPort + index,
                        EngineRpcService.StatusVersionPB.getDefaultInstance(), 1000).get(2, TimeUnit.SECONDS).getAlive());
            }
            int expectedChannelCount = statusPort == httpPort + 1 ? 6 : 6 + engineCount;
            assertEquals(expectedChannelCount, client.created.size());
            for (int refresh = 0; refresh < 3; refresh++) {
                client.onAddressUpdate(hosts);

                assertFalse(client.created.stream().anyMatch(ManagedChannel::isShutdown));
                for (int index = 0; index < engineCount; index++) {
                    assertTrue(client.getWorkerStatusAsync("10.0.0.1", statusPort + index,
                            EngineRpcService.StatusVersionPB.getDefaultInstance(), 1000).get(2, TimeUnit.SECONDS).getAlive());
                    assertEquals(1, Collections.frequency(client.createdKeys,
                            "10.0.0.1:" + (statusPort + index) + ":worker"));
                }
                assertEquals(expectedChannelCount, client.created.size());
            }

            client.onAddressUpdate(List.of());
            assertTrue(client.created.stream().allMatch(ManagedChannel::isShutdown));
        } finally {
            client.shutdownChannelPool();
            executor.shutdownNow();
            server.shutdownNow().awaitTermination(2, TimeUnit.SECONDS);
        }
    }

    private static final class RecordingClient extends EngineGrpcClient {
        private final String serverName;
        private final List<ManagedChannel> created = new ArrayList<>();
        private final List<String> createdKeys = new ArrayList<>();

        private RecordingClient(String serverName, ThreadPoolExecutor executor) {
            super(listener -> { }, executor, mock(EventLoopGroup.class), mock(GrpcReporter.class), 20);
            this.serverName = serverName;
        }

        @Override
        protected ManagedChannel createChannel(String channelKey) {
            ManagedChannel channel = InProcessChannelBuilder.forName(serverName).directExecutor().build();
            created.add(channel);
            createdKeys.add(channelKey);
            return channel;
        }
    }
}
