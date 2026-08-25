package org.flexlb.dao.optimizer;

import org.flexlb.util.JsonUtils;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class OptimizerProtocolDtoTest {

    @Test
    void deserializesTraceQueryResponse() {
        String json = """
                {
                  "header": {"status": {"code": "OK"}},
                  "total_blocks": "10",
                  "capacity_results": [{
                    "capacity_gb": 1.5,
                    "cache_hit_count": "6",
                    "hit_rate": 0.75,
                    "current_unique_keys": "100"
                  }],
                  "theoretical_result": {
                    "max_hit_count": "8",
                    "current_unique_keys": "120",
                    "hit_rate": 1.0
                  },
                  "input_token_len": "128"
                }
                """;

        OptimizerTraceQueryResponse response = JsonUtils.toObject(json, OptimizerTraceQueryResponse.class);

        assertEquals(OptimizerErrorCode.OK, response.getHeader().getStatus().getCode());
        assertEquals(10L, response.getTotalBlocks());
        assertEquals(128L, response.getInputTokenLen());
        assertEquals(1.5, response.getCapacityResults().getFirst().getCapacityGb());
        assertEquals(6L, response.getCapacityResults().getFirst().getCacheHitCount());
        assertEquals(0.75, response.getCapacityResults().getFirst().getHitRate());
        assertEquals(100L, response.getCapacityResults().getFirst().getCurrentUniqueKeys());
        assertEquals(8L, response.getTheoreticalResult().getMaxHitCount());
        assertEquals(120L, response.getTheoreticalResult().getCurrentUniqueKeys());
        assertEquals(1.0, response.getTheoreticalResult().getHitRate());
    }

    @Test
    void mapsUnknownTraceQueryErrorCodesToUnknownError() {
        OptimizerTraceQueryResponse namedResponse = JsonUtils.toObject("""
                {"header":{"status":{"code":"NODE_NOT_REGISTERED"}}}
                """, OptimizerTraceQueryResponse.class);
        OptimizerTraceQueryResponse numericResponse = JsonUtils.toObject("""
                {"header":{"status":{"code":999}}}
                """, OptimizerTraceQueryResponse.class);

        assertEquals(OptimizerErrorCode.UNKNOWN_ERROR, namedResponse.getHeader().getStatus().getCode());
        assertEquals(OptimizerErrorCode.UNKNOWN_ERROR, numericResponse.getHeader().getStatus().getCode());
    }
}
