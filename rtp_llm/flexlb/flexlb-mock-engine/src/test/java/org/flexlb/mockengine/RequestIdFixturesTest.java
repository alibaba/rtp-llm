package org.flexlb.mockengine;

import org.flexlb.engine.grpc.EngineRpcService;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

class RequestIdFixturesTest {

    @Test
    void writesCanonicalEngineIdsIncludingZero() {
        for (String requestId : new String[]{"0", "-1", Long.toString(Long.MAX_VALUE)}) {
            EngineRpcService.GenerateInputPB input = RequestIdFixtures.write(
                    EngineRpcService.GenerateInputPB.newBuilder(), requestId).build();
            assertEquals(Long.parseLong(requestId), input.getRequestId());
            assertTrue(input.getUnknownFields().asMap().isEmpty());
        }
    }

    @Test
    void rejectsIdsOutsideTheEngineInt64Contract() {
        for (String requestId : new String[]{"", "request-a", "01", "+1", "-0", "9223372036854775808"}) {
            assertThrows(IllegalArgumentException.class, () -> RequestIdFixtures.write(
                    EngineRpcService.GenerateInputPB.newBuilder(), requestId));
        }
    }
}
