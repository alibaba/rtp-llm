package org.flexlb.mockengine;

import java.util.concurrent.atomic.AtomicLong;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import java.nio.file.Path;
import java.nio.file.Files;
import static org.junit.jupiter.api.Assertions.*;

class ExpiringRequestIdsTest {
    @TempDir Path directory;

    private MockPerformanceModel model() throws Exception {
        Path master = directory.resolve("master.json"), performance = directory.resolve("performance.json");
        MockMasterConfig.writeWithPrefillExpression(master, "1 + sum(computeTokens)");
        Files.writeString(performance, "{}");
        return MockPerformanceModel.load(performance.toString(), master.toString());
    }

    @Test void livePrefillFenceSurvivesMoreThanTenThousandNewFences() throws Exception {
        try (var cluster = MockEngineTestCluster.create(model(), 63300, 1, 0)) {
            var prefill = cluster.prefill(0);
            for (long id = 1; id <= 10_001; id++) prefill.cancelRequest(id);
            var response = MockEngineTestSupport.enqueue(prefill, MockEngineTestSupport.batch(1,
                    MockEngineTestSupport.slot(0, MockEngineTestSupport.input(1, 5))));
            assertEquals(0, response.getSuccessesCount());
            assertEquals(8429, response.getErrors(0).getErrorInfo().getErrorCode());
            assertEquals(0, prefill.getInflightCount());
        }
    }

    @Test void unsupportedOutputWidthIsRejectedBeforeAnyAdmission() throws Exception {
        try (var cluster = MockEngineTestCluster.create(model(), 63300, 1, 0)) {
            var prefill = cluster.prefill(0);
            var input = MockEngineTestSupport.input(1, 5).toBuilder();
            input.setGenerateConfig(input.getGenerateConfig().toBuilder().setNumReturnSequences(3));
            var response = MockEngineTestSupport.enqueue(prefill, MockEngineTestSupport.batch(1,
                    MockEngineTestSupport.slot(0, input.build())));
            assertEquals(0, response.getSuccessesCount());
            assertEquals(io.grpc.Status.Code.UNIMPLEMENTED.value(), response.getErrors(0).getErrorInfo().getErrorCode());
            var error = new java.util.concurrent.atomic.AtomicReference<Throwable>();
            prefill.generateStreamCall(input.build(), new io.grpc.stub.StreamObserver<>() {
                public void onNext(org.flexlb.engine.grpc.EngineRpcService.GenerateOutputsPB value) { fail("unsupported request executed"); }
                public void onError(Throwable failure) { error.set(failure); }
                public void onCompleted() { fail("unsupported request completed"); }
            });
            assertEquals(io.grpc.Status.Code.UNIMPLEMENTED, io.grpc.Status.fromThrowable(error.get()).getCode());
            assertEquals(0, prefill.getInflightCount());
        }
    }
    @Test void historySurvivesDiagnosticCapsAndExpiresAtItsOwnDeadline() {
        AtomicLong now = new AtomicLong();
        var history = new ExpiringRequestIds(now::get, 600);
        for (long id = 0; id <= 20_000; id++) history.add(id);
        assertTrue(history.contains(0));
        now.set(599);
        assertTrue(history.contains(0));
        now.set(600);
        assertFalse(history.contains(0));
    }
    @Test void refreshedIdsSurviveStaleExpiryEntriesAndCrashClearsEverything() {
        AtomicLong now = new AtomicLong();
        var history = new ExpiringRequestIds(now::get, 600);
        history.add(1);
        now.set(300); history.add(1);
        now.set(600); assertTrue(history.contains(1));
        now.set(900); assertFalse(history.contains(1));
        history.add(2); history.clear(); assertFalse(history.contains(2));
    }
    @Test void multiSequenceAndBeamInputsAreExplicitlyUnsupported() {
        var input = MockEngineTestSupport.input(1, 5);
        assertTrue(JavaMockEngineCluster.FastRpcService.supportsSingleSequence(input));
        var config = input.getGenerateConfig().toBuilder().setNumReturnSequences(3);
        assertFalse(JavaMockEngineCluster.FastRpcService.supportsSingleSequence(input.toBuilder().setGenerateConfig(config).build()));
        config.setNumReturnSequences(1).setNumBeams(4);
        assertFalse(JavaMockEngineCluster.FastRpcService.supportsSingleSequence(input.toBuilder().setGenerateConfig(config).build()));
        config.setNumBeams(1).addVariableNumBeams(1).addVariableNumBeams(4);
        assertFalse(JavaMockEngineCluster.FastRpcService.supportsSingleSequence(input.toBuilder().setGenerateConfig(config).build()));
    }
}
