package org.flexlb.balance.scheduler;

import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CancelReasonTest {
    @ParameterizedTest
    @ValueSource(strings = {"007", "+7", "-0", "request-a", "9223372036854775808"})
    void invalidIdsCannotReachOrCancelAnotherEngineRequest(String requestId) {
        assertThrows(
                IllegalArgumentException.class,
                () -> CancelReason.PRIORITY_PREEMPTED.toRequest(requestId));
    }

    @ParameterizedTest
    @ValueSource(longs = {0L, 7L, 9007199254740993L, Long.MIN_VALUE, Long.MAX_VALUE})
    void sendsTheExactEngineInt64IdAtTheCancellationBoundary(long requestId) throws Exception {
        var request = CancelReason.PRIORITY_PREEMPTED.toRequest(Long.toString(requestId));
        var received = EngineRpcService.CancelRequestPB.parseFrom(request.toByteString());
        assertEquals(requestId, received.getRequestId());
        assertEquals(
                EngineRpcService.RequestCancelReasonPB.REQUEST_CANCEL_REASON_PRIORITY_PREEMPTED,
                received.getReason());
    }
}
