package org.flexlb.interceptor;

import io.grpc.Context;
import io.grpc.Metadata;
import io.grpc.ServerInterceptors;
import io.grpc.netty.NettyChannelBuilder;
import io.grpc.netty.NettyServerBuilder;
import io.grpc.stub.StreamObserver;
import org.flexlb.schedule.grpc.FlexlbScheduleProtocol.FlexlbScheduleRequestPB;
import org.flexlb.schedule.grpc.FlexlbScheduleProtocol.FlexlbScheduleResponsePB;
import org.flexlb.schedule.grpc.FlexlbServiceGrpc;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;

class GrpcRequestMessageSizeTest {

    @Test
    void transportMeasuresUncompressedBytesAndIsolatesConcurrentCalls() throws Exception {
        var interceptor = new GrpcServerTimingInterceptor();
        Map<String, Long> measuredSizes = new ConcurrentHashMap<>();
        var service = new FlexlbServiceGrpc.FlexlbServiceImplBase() {
            @Override
            public void schedule(FlexlbScheduleRequestPB request,
                                 StreamObserver<FlexlbScheduleResponsePB> observer) {
                measuredSizes.put(request.getRequestId(), GrpcServerTimingInterceptor.getRequestMessageBytes());
                assertNotNull(GrpcServerTimingInterceptor.getNanos());
                observer.onNext(FlexlbScheduleResponsePB.newBuilder().setSuccess(true).build());
                observer.onCompleted();
            }
        };
        var server = NettyServerBuilder.forPort(0)
                .addStreamTracerFactory(interceptor.requestSizeTracerFactory())
                .addService(ServerInterceptors.intercept(service, interceptor)).build().start();
        var channel = NettyChannelBuilder.forAddress("127.0.0.1", server.getPort()).usePlaintext().build();
        try {
            var stub = FlexlbServiceGrpc.newFutureStub(channel).withDeadlineAfter(5, TimeUnit.SECONDS);
            var pending = new ArrayList<com.google.common.util.concurrent.ListenableFuture<FlexlbScheduleResponsePB>>();
            var requests = new ArrayList<FlexlbScheduleRequestPB>();
            for (int i = 0; i < 20; i++) {
                var request = FlexlbScheduleRequestPB.newBuilder()
                        .setRequestId("request-" + i + "-" + "x".repeat(i * 300))
                        .build();
                requests.add(request);
                pending.add((i % 2 == 0 ? stub : stub.withCompression("gzip")).schedule(request));
            }
            for (var future : pending) {
                future.get(5, TimeUnit.SECONDS);
            }
            for (var request : requests) {
                assertEquals((long) request.getSerializedSize(), measuredSizes.get(request.getRequestId()));
            }
            var empty = FlexlbScheduleRequestPB.getDefaultInstance();
            stub.schedule(empty).get(5, TimeUnit.SECONDS);
            assertEquals(0L, measuredSizes.get(""));
            assertNull(GrpcServerTimingInterceptor.getRequestMessageBytes());
        } finally {
            channel.shutdownNow().awaitTermination(3, TimeUnit.SECONDS);
            server.shutdownNow().awaitTermination(3, TimeUnit.SECONDS);
        }
    }

    @Test
    void unknownSizeDoesNotBecomeAZeroSample() {
        var tracer = new GrpcServerTimingInterceptor().requestSizeTracerFactory()
                .newServerStreamTracer(FlexlbServiceGrpc.getScheduleMethod().getFullMethodName(), new Metadata());
        var context = tracer.filterContext(Context.ROOT);
        context.run(() -> assertNull(GrpcServerTimingInterceptor.getRequestMessageBytes()));
        tracer.inboundMessageRead(0, 17, -1);
        tracer.inboundUncompressedSize(100);
        tracer.inboundUncompressedSize(200);
        context.run(() -> assertEquals(300L, GrpcServerTimingInterceptor.getRequestMessageBytes()));
        assertNull(GrpcServerTimingInterceptor.getRequestMessageBytes());
    }
}
