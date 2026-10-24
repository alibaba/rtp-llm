package org.flexlb.mockengine;

import io.grpc.Status;
import io.grpc.stub.StreamObserver;
import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.LinkedBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import static org.flexlb.mockengine.MockEngineTestSupport.*;
import static org.junit.jupiter.api.Assertions.*;

/** Both public RPCs must preserve the real engine domain-error contract. */
class ResponsePumpParityTest {
    @TempDir Path directory;

    private static final class Output implements StreamObserver<EngineRpcService.GenerateOutputsPB> {
        final List<EngineRpcService.GenerateOutputsPB> frames = new CopyOnWriteArrayList<>();
        final CountDownLatch done = new CountDownLatch(1);
        final AtomicInteger terminals = new AtomicInteger();
        volatile Throwable error;
        public void onNext(EngineRpcService.GenerateOutputsPB output) { frames.add(output); }
        public void onError(Throwable error) { this.error = error; terminals.incrementAndGet(); done.countDown(); }
        public void onCompleted() { terminals.incrementAndGet(); done.countDown(); }
    }

    private static EngineRpcService.GenerateOutputsPB frame(boolean finished) {
        return EngineRpcService.GenerateOutputsPB.newBuilder().setRequestId(71)
                .setFlattenOutput(EngineRpcService.FlattenOutputPB.newBuilder().addFinished(finished)).build();
    }

    private Output start(JavaMockEngineCluster.FastRpcService service, boolean fetch) {
        Output output = new Output();
        if (fetch) service.fetchResponse(EngineRpcService.FetchRequestPB.newBuilder().setRequestId(71).build(), output);
        else service.generateStreamCall(inputWithDecode(71, 16, service.getGrpcPort(), 8), output);
        return output;
    }

    @ParameterizedTest
    @CsvSource({"false,0", "true,0", "false,602", "true,602", "false,603", "true,603",
            "false,8100", "true,8100", "false,8429", "true,8429"})
    void firstTokenThenTerminalPreservesFramesAndErrorDetails(boolean fetch, int errorCode) throws Exception {
        try (var cluster = MockEngineTestCluster.create(performanceModel(directory, "5000", 1, 5000), 62100, 0, 1)) {
            var service = cluster.decode(0);
            var queue = new LinkedBlockingQueue<EngineRpcService.GenerateOutputsPB>();
            queue.add(frame(false));
            var terminal = errorCode == 0 ? frame(true) : EngineRpcService.GenerateOutputsPB.newBuilder()
                    .setRequestId(71).setErrorInfo(EngineRpcService.RpcErrorPB.newBuilder()
                            .setErrorCodeValue(errorCode).setErrorMessage("terminal error")).build();
            queue.add(terminal);
            var afterTerminal = frame(false);
            queue.add(afterTerminal);
            service.requests.output(71, queue);
            var output = start(service, fetch);
            assertTrue(output.done.await(3, TimeUnit.SECONDS));
            assertEquals(1, output.terminals.get());
            if (errorCode != 0) {
                var expectedStatus = switch (errorCode) {
                    case 603 -> Status.Code.DEADLINE_EXCEEDED;
                    case 8100 -> Status.Code.CANCELLED;
                    default -> Status.Code.RESOURCE_EXHAUSTED;
                };
                assertEquals(expectedStatus,
                        Status.fromThrowable(output.error).getCode());
                var details = MockRpcErrors.downstream(output.error);
                assertEquals(errorCode, details.getErrorCode());
                assertEquals("terminal error", details.getErrorMessage());
                assertEquals(List.of(frame(false)), output.frames);
            } else {
                assertNull(output.error);
                assertEquals(List.of(frame(false), terminal), output.frames);
            }
            assertSame(afterTerminal, queue.peek(), "never consume frames after a terminal");
            service.cancel(71);
            cluster.awaitAllInflightZero(7000);
            assertEquals(0, service.requests.outputCount());
            assertNull(service.requests.deadline(71));
        }
    }

    @ParameterizedTest
    @CsvSource({"false,false", "true,false", "false,true", "true,true"})
    void frameGapTimeoutCompletesWithExactlyTheAvailableFrames(boolean fetch, boolean firstToken) throws Exception {
        try (var cluster = MockEngineTestCluster.create(performanceModel(directory, "5000", 1, 5000), 62100, 0, 1)) {
            var service = cluster.decode(0);
            service.setResponsePollTimeoutMs(30);
            var queue = new LinkedBlockingQueue<EngineRpcService.GenerateOutputsPB>();
            if (firstToken) queue.add(frame(false));
            service.requests.output(71, queue);
            var output = start(service, fetch);
            assertTrue(output.done.await(3, TimeUnit.SECONDS));
            assertNull(output.error);
            assertEquals(firstToken ? List.of(frame(false)) : List.of(), output.frames);
            assertEquals(1, output.terminals.get());
            service.cancel(71);
            cluster.awaitAllInflightZero(7000);
        }
    }
}
