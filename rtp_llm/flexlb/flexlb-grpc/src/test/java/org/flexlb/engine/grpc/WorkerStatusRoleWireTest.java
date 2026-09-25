package org.flexlb.engine.grpc;

import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

import java.util.HexFormat;

import static org.junit.jupiter.api.Assertions.assertEquals;

class WorkerStatusRoleWireTest {

    // 使用 DashLLM model_rpc_service_pb2.WorkerStatusPB.SerializeToString() 生成，
    // 覆盖双写、旧字符串和仅枚举，避免用 FlexLB 自己的 writer 验证自己。
    @ParameterizedTest
    @CsvSource({
            "PREFILL, 0a10526f6c65547970652e50524546494c4ca00101",
            "PREFILL, 0a10526f6c65547970652e50524546494c4c",
            "PREFILL, a00101",
            "DECODE, 0a0f526f6c65547970652e4445434f4445a00102",
            "DECODE, 0a0f526f6c65547970652e4445434f4445",
            "DECODE, a00102",
            "PDFUSION, 0a11526f6c65547970652e5044465553494f4e"
    })
    void preservesDashLlmRoleAcrossWireParsing(RoleType expected, String hex) throws Exception {
        EngineRpcService.WorkerStatusPB response = EngineRpcService.WorkerStatusPB.parseFrom(
                HexFormat.of().parseHex(hex));

        assertEquals(expected, RoleTypeProtoConverter.fromWorkerStatus(response));
    }
}
