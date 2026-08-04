package org.flexlb.engine.grpc.client;

import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import io.grpc.Server;
import io.grpc.ServerBuilder;
import io.grpc.Status;
import io.grpc.StatusRuntimeException;
import io.grpc.stub.StreamObserver;
import org.flexlb.engine.grpc.core.GrpcChannelFactory;
import org.flexlb.engine.grpc.core.GrpcTarget;
import org.flexlb.engine.grpc.monitor.GrpcRuntimeMetrics;
import org.flexlb.kvcm.grpc.GetClusterInfoRequest;
import org.flexlb.kvcm.grpc.MetaServiceGrpc;
import org.junit.jupiter.api.Test;
import org.mockito.Mockito;

import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.mockito.Mockito.when;

class KvcmMetaServiceClientTest {

    @Test
    void propagatesClusterInfoDeadline() throws Exception {
        Server server = ServerBuilder.forPort(0)
                .addService(new MetaServiceGrpc.MetaServiceImplBase() {
                    @Override
                    public void getClusterInfo(
                            GetClusterInfoRequest request,
                            StreamObserver<org.flexlb.kvcm.grpc.GetClusterInfoResponse> responseObserver) {
                    }
                })
                .build()
                .start();
        ManagedChannel channel = ManagedChannelBuilder.forAddress("127.0.0.1", server.getPort())
                .usePlaintext()
                .build();
        KvcmMetaServiceClient client = null;
        try {
            GrpcChannelFactory channelFactory = Mockito.mock(GrpcChannelFactory.class);
            GrpcTarget target = new GrpcTarget("127.0.0.1", server.getPort());
            when(channelFactory.create(target)).thenReturn(channel);
            client = new KvcmMetaServiceClient(channelFactory, Mockito.mock(GrpcRuntimeMetrics.class));
            KvcmMetaServiceClient metaServiceClient = client;

            StatusRuntimeException exception = assertThrows(StatusRuntimeException.class,
                    () -> metaServiceClient.getClusterInfo(
                            target,
                            GetClusterInfoRequest.getDefaultInstance(),
                            100));

            assertEquals(Status.Code.DEADLINE_EXCEEDED, exception.getStatus().getCode());
        } finally {
            close(client, server, channel);
        }
    }

    private static void close(KvcmMetaServiceClient client, Server server, ManagedChannel channel)
            throws InterruptedException {
        if (client != null) {
            client.shutdown();
        } else {
            channel.shutdownNow();
        }
        channel.awaitTermination(1, TimeUnit.SECONDS);
        server.shutdownNow();
        server.awaitTermination(1, TimeUnit.SECONDS);
    }
}
