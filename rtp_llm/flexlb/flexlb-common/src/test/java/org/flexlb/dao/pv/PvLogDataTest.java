package org.flexlb.dao.pv;

import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.util.JsonUtils;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class PvLogDataTest {

    @Test
    void includesBlockHashAndKvcmTimings() {
        Request request = new Request();
        request.setRequestId("request-1");
        request.setSeqLen(128);
        request.setRequestTimeMs(1000);

        BalanceContext context = new BalanceContext();
        context.setStartTime(1500);
        context.setRequest(request);
        context.setResponse(new Response());
        context.recordRequestTiming(request.getRequestTimeMs(), 9);
        context.recordBlockHashTiming(12, 34);
        context.recordKvcmQuery(56);
        context.recordKvcmQuery(78);

        PvLogData data = new PvLogData(context);

        assertEquals(500, data.getArrivalMs());
        assertEquals(9, data.getDecodeUs());
        assertEquals(12, data.getHashWaitUs());
        assertEquals(34, data.getHashUs());
        assertEquals(134, data.getKvcmUs());
        assertEquals(2, data.getKvcmCount());

        String json = JsonUtils.toStringOrEmpty(data);
        assertTrue(json.contains("\"arrivalMs\":500"));
        assertTrue(json.contains("\"decodeUs\":9"));
        assertTrue(json.contains("\"hashWaitUs\":12"));
        assertTrue(json.contains("\"hashUs\":34"));
        assertTrue(json.contains("\"kvcmUs\":134"));
        assertTrue(json.contains("\"kvcmCount\":2"));
    }
}
