package org.flexlb.mockengine;

import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.api.Test;
import java.util.List;
import static org.junit.jupiter.api.Assertions.*;

class JavaLoadClientStreamCompletionTest {
    private static EngineRpcService.GenerateOutputsPB frame(boolean finished) {
        return EngineRpcService.GenerateOutputsPB.newBuilder()
                .setFlattenOutput(EngineRpcService.FlattenOutputPB.newBuilder().addFinished(finished))
                .build();
    }

    private static JavaLoadClient.RequestResult consume(EngineRpcService.GenerateOutputsPB... frames) {
        JavaLoadClient.RequestResult result = new JavaLoadClient.RequestResult();
        JavaLoadClient.consumeStream(List.of(frames).iterator(), result, System.nanoTime());
        return result;
    }

    @Test void nonemptyEofIsNotSuccess() {
        assertEquals("incomplete_response", consume(frame(false)).status);
    }
    @Test void emptyEofRemainsAnError() {
        assertEquals("empty_response", consume().status);
    }
    @Test void finishedIsBusinessSuccess() {
        assertEquals("ok", consume(frame(false), frame(true)).status);
    }
    @Test void typedErrorOverridesEvenAFinishedFrame() {
        EngineRpcService.GenerateOutputsPB error = frame(true).toBuilder()
                .setErrorInfo(EngineRpcService.RpcErrorPB.newBuilder()
                        .setErrorCodeValue(8211).setErrorMessage("lack memory")).build();
        JavaLoadClient.RequestResult result = consume(frame(false), error);
        assertEquals("engine_error", result.status);
        assertEquals("business", result.errorKind);
        assertTrue(result.error.contains("8211"));
    }
}
