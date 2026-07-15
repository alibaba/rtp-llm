package org.flexlb.dao.pv;

import org.flexlb.dao.BalanceContext;
import org.flexlb.dao.loadbalance.Request;
import org.flexlb.dao.loadbalance.Response;
import org.flexlb.dao.route.RoleType;
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
        context.recordCacheMatch("KVCM", 56, RoleType.PREFILL, "10.0.0.1", 256);
        context.recordCacheMatch("KVCM", 78, RoleType.PREFILL, "10.0.0.2", 512);
        context.recordCacheMatch("KVCM", 10, RoleType.DECODE, "10.0.0.3", 128);
        context.finishRequestTiming();

        PvLogData data = new PvLogData(context);

        assertEquals(context.getTotalTimeUs(), data.getTotalUs());
        assertEquals(500, data.getArrivalMs());
        assertEquals(9, data.getReqParseUs());
        assertEquals(12, data.getHashWaitUs());
        assertEquals(34, data.getHashUs());
        assertEquals("KVCM", data.getCacheMatchSource());
        assertEquals(144, data.getCacheMatchUs());
        assertEquals(3, data.getCacheMatchCount());
        assertEquals(2, data.getCacheMatchSelections().size());
        assertEquals("10.0.0.2", data.getCacheMatchSelections().getFirst().selectedIp());
        assertEquals(512, data.getCacheMatchSelections().getFirst().hitCacheTokens());

        String json = JsonUtils.toStringOrEmpty(data);
        assertTrue(json.contains("\"totalUs\":" + context.getTotalTimeUs()));
        assertTrue(json.contains("\"arrivalMs\":500"));
        assertTrue(json.contains("\"reqParseUs\":9"));
        assertTrue(json.contains("\"hashWaitUs\":12"));
        assertTrue(json.contains("\"hashUs\":34"));
        assertTrue(json.contains("\"cacheMatchSource\":\"KVCM\""));
        assertTrue(json.contains("\"cacheMatchUs\":144"));
        assertTrue(json.contains("\"cacheMatchCount\":3"));
        assertTrue(json.contains("\"cacheMatchSelections\":[{\"role\":\"PREFILL\",\"selectedIp\":\"10.0.0.2\",\"hitCacheTokens\":512}"));
    }
}
