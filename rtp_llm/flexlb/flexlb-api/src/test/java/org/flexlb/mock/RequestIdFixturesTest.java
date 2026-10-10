package org.flexlb.mock;

import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class RequestIdFixturesTest {

    @Test
    void writesCanonicalInt64IdsIncludingZero() {
        for (String requestId : new String[] {"0", "-1", Long.toString(Long.MAX_VALUE)}) {
            EngineRpcService.EnqueueBatchSuccessPB success = RequestIdFixtures.write(
                    EngineRpcService.EnqueueBatchSuccessPB.newBuilder(), requestId).build();
            assertEquals(Long.parseLong(requestId), success.getRequestId());
            assertEquals(0, success.getUnknownFields().asMap().size());
        }
    }

    @Test
    void rejectsIdsThatCannotBeRepresentedByTheEngineField() {
        for (String requestId : new String[] {"", "request-a", "01", "+1", "-0", "9223372036854775808"}) {
            assertThrows(IllegalArgumentException.class, () -> RequestIdFixtures.write(
                    EngineRpcService.EnqueueBatchErrorPB.newBuilder(), requestId));
        }
    }
}
