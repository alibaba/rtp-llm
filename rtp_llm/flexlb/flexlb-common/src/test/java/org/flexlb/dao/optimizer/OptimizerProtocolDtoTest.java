package org.flexlb.dao.optimizer;

import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.Test;

import java.util.List;

class OptimizerProtocolDtoTest {

    private final ObjectMapper objectMapper = new ObjectMapper();

    @Test
    void serializesLatestRegisterRequestShapeAndInt64LocationSize() throws Exception {
        OptimizerRegisterRequest request = new OptimizerRegisterRequest();
        request.setTraceId("trace-1");
        request.setInstanceGroup("group-a");
        request.setInstanceId("instance-a");
        request.setBlockSize(16);
        request.setLocationSpecInfos(List.of(
                new OptimizerRegisterRequest.LocationSpecInfo("full", 4_294_967_296L),
                new OptimizerRegisterRequest.LocationSpecInfo("linear", 65_536L)));
        request.setLocationSpecGroups(List.of(
                new OptimizerRegisterRequest.LocationSpecGroup("full-group", List.of("full")),
                new OptimizerRegisterRequest.LocationSpecGroup("linear-group", List.of("linear"))));
        request.setOptimizerStateInfo(new OptimizerStateInfo("full-group", "linear-group"));
        request.setLinearStep(4);

        JsonNode json = objectMapper.readTree(objectMapper.writeValueAsString(request));

        Assertions.assertEquals(4_294_967_296L, json.at("/location_spec_infos/0/size").longValue());
        Assertions.assertEquals(
                "full-group",
                json.at("/optimizer_state_info/full_location_spec_group_name").textValue());
        Assertions.assertEquals(
                "linear-group",
                json.at("/optimizer_state_info/linear_location_spec_group_name").textValue());
        Assertions.assertFalse(json.has("full_group_name"));
    }

    @Test
    void deserializesLatestRegisterResponse() throws Exception {
        String json = """
                {
                  "header": {
                    "status": {"code": "OK", "message": "success"},
                    "request_id": "request-1",
                    "tracer_result": "trace-result"
                  },
                  "estimated_capacity_blocks": ["123", "456"],
                  "size_full_only": "4294967296",
                  "size_full_linear": "8589934592"
                }
                """;

        OptimizerRegisterResponse response = objectMapper.readValue(json, OptimizerRegisterResponse.class);

        Assertions.assertTrue(response.getHeader().getStatus().isOk());
        Assertions.assertEquals(OptimizerErrorCode.OK, response.getHeader().getStatus().getCode());
        Assertions.assertEquals("request-1", response.getHeader().getRequestId());
        Assertions.assertEquals("trace-result", response.getHeader().getTracerResult());
        Assertions.assertEquals(List.of(123L, 456L), response.getEstimatedCapacityBlocks());
        Assertions.assertEquals(4_294_967_296L, response.getSizeFullOnly());
        Assertions.assertEquals(8_589_934_592L, response.getSizeFullLinear());
    }

    @Test
    void deserializesLatestGetResponseAndMatchesNestedOptimizerState() throws Exception {
        String json = """
                {
                  "header": {"status": {"code": "OK"}},
                  "instance_group": "group-a",
                  "instance_id": "instance-a",
                  "block_size": 16,
                  "location_spec_infos": [
                    {"name": "full", "size": "4294967296"},
                    {"name": "linear", "size": "65536"}
                  ],
                  "location_spec_groups": [
                    {"name": "full-group", "spec_names": ["full"]},
                    {"name": "linear-group", "spec_names": ["linear"]}
                  ],
                  "optimizer_state_info": {
                    "full_location_spec_group_name": "full-group",
                    "linear_location_spec_group_name": "linear-group"
                  },
                  "linear_step": 4
                }
                """;
        OptimizerGetInstanceResponse response = objectMapper.readValue(json, OptimizerGetInstanceResponse.class);
        OptimizerInstanceParams params = OptimizerInstanceParams.builder()
                .instanceGroup("group-a")
                .blockSize(16)
                .locationSpecInfos(List.of(
                        new OptimizerRegisterRequest.LocationSpecInfo("full", 4_294_967_296L),
                        new OptimizerRegisterRequest.LocationSpecInfo("linear", 65_536L)))
                .locationSpecGroups(List.of(
                        new OptimizerRegisterRequest.LocationSpecGroup("linear-group", List.of("linear")),
                        new OptimizerRegisterRequest.LocationSpecGroup("full-group", List.of("full"))))
                .optimizerStateInfo(new OptimizerStateInfo("full-group", "linear-group"))
                .linearStep(4)
                .build();

        Assertions.assertEquals(4_294_967_296L, response.getLocationSpecInfos().getFirst().getSize());
        Assertions.assertTrue(params.matchesRemote(response));

        params.setOptimizerStateInfo(new OptimizerStateInfo("full-group", "different-linear-group"));
        Assertions.assertFalse(params.matchesRemote(response));
    }

    @Test
    void deserializesLatestTraceQueryResponse() throws Exception {
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

        OptimizerTraceQueryResponse response = objectMapper.readValue(json, OptimizerTraceQueryResponse.class);

        Assertions.assertEquals(10L, response.getTotalBlocks());
        Assertions.assertEquals(128L, response.getInputTokenLen());
        Assertions.assertEquals(1.5, response.getCapacityResults().getFirst().getCapacityGb());
        Assertions.assertEquals(6L, response.getCapacityResults().getFirst().getCacheHitCount());
        Assertions.assertEquals(0.75, response.getCapacityResults().getFirst().getHitRate());
        Assertions.assertEquals(100L, response.getCapacityResults().getFirst().getCurrentUniqueKeys());
        Assertions.assertEquals(8L, response.getTheoreticalResult().getMaxHitCount());
        Assertions.assertEquals(120L, response.getTheoreticalResult().getCurrentUniqueKeys());
        Assertions.assertEquals(1.0, response.getTheoreticalResult().getHitRate());
    }
}
