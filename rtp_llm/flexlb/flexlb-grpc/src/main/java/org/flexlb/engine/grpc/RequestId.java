package org.flexlb.engine.grpc;

import com.google.protobuf.ByteString;
import com.google.protobuf.InvalidProtocolBufferException;
import com.google.protobuf.UnknownFieldSet;
import org.flexlb.schedule.grpc.FlexlbScheduleProtocol;

import java.util.List;

/**
 * 在 Protobuf 边界统一读取请求 ID，业务层始终使用字符串，不生成或映射新的 ID。
 *
 * <p>通过具体 PB 类型生成的 getter 读取字段，不使用描述符或反射。
 * string 与 int64 的编码类型不同，旧编码保留在同编号的 unknown fields 中，兼容读取时不需要第二个 ID 字段。
 */
public final class RequestId {
    private RequestId() {
    }

    /**
     * 读取 Worker Status 的字符串 ID，并兼容同编号的整数编码。
     */
    public static String parse(EngineRpcService.TaskInfoPBOrBuilder message) {
        return parseString(message.getRequestId(), message.getUnknownFields(),
                EngineRpcService.TaskInfoPB.REQUEST_ID_FIELD_NUMBER);
    }

    /** 直接读取字符串 ID；缺失时兼容同编号的旧整数编码。OrBuilder 同时支持消息和 Builder。 */
    public static String parse(FlexlbScheduleProtocol.FlexlbScheduleRequestPBOrBuilder message) {
        return parseString(message.getRequestId(), message.getUnknownFields(), FlexlbScheduleProtocol.FlexlbScheduleRequestPB.REQUEST_ID_FIELD_NUMBER);
    }

    /** 直接读取字符串 ID；缺失时兼容同编号的旧整数编码。OrBuilder 同时支持消息和 Builder。 */
    public static String parse(FlexlbScheduleProtocol.FlexlbCancelRequestPBOrBuilder message) {
        return parseString(message.getRequestId(), message.getUnknownFields(), FlexlbScheduleProtocol.FlexlbCancelRequestPB.REQUEST_ID_FIELD_NUMBER);
    }

    /** 直接读取字符串 ID；缺失时兼容同编号的旧整数编码。OrBuilder 同时支持消息和 Builder。 */
    public static String parse(FlexlbScheduleProtocol.GetRequestStateRequestPBOrBuilder message) {
        return parseString(message.getRequestId(), message.getUnknownFields(), FlexlbScheduleProtocol.GetRequestStateRequestPB.REQUEST_ID_FIELD_NUMBER);
    }

    /** 将引擎消息中的 int64 请求 ID 转为字符串。 */
    public static String parse(EngineRpcService.GenerateInputPBOrBuilder message) {
        return Long.toString(message.getRequestId());
    }

    /** 将引擎消息中的 int64 请求 ID 转为字符串。 */
    public static String parse(EngineRpcService.EnqueueBatchSuccessPBOrBuilder message) {
        return Long.toString(message.getRequestId());
    }

    /** 将引擎消息中的 int64 请求 ID 转为字符串。 */
    public static String parse(EngineRpcService.EnqueueBatchErrorPBOrBuilder message) {
        return Long.toString(message.getRequestId());
    }

    /**
     * Validate the current Engine's int64 generation boundary before admitting a batch request.
     * The Master keeps the original string ID; it does not normalize or remap it.
     */
    public static void requireMatchingGenerateInput(String requestId, ByteString generateInput) {
        if (generateInput.isEmpty()) {
            throw new IllegalArgumentException("generate_input is required for BATCH generation");
        }
        EngineRpcService.GenerateInputPB input;
        try {
            input = EngineRpcService.GenerateInputPB.parseFrom(generateInput);
        } catch (InvalidProtocolBufferException error) {
            throw new IllegalArgumentException("generate_input must contain a valid GenerateInputPB", error);
        }
        if (!parse(input).equals(requestId)) {
            throw new IllegalArgumentException(
                    "BATCH request_id must be the canonical int64 decimal string matching generate_input.request_id");
        }
    }

    /**
     * Encode an Engine cancellation ID without allowing a different spelling to cancel another Master ID.
     */
    public static long toEngineRequestId(String requestId) {
        long engineRequestId = Long.parseLong(requestId);
        if (!Long.toString(engineRequestId).equals(requestId)) {
            throw new IllegalArgumentException("Engine request_id must be a canonical int64 decimal string");
        }
        return engineRequestId;
    }

    /** 优先返回非空白字符串 ID，否则兼容同编号的旧整数编码。 */
    private static String parseString(String value, UnknownFieldSet unknown, int fieldNumber) {
        if (!value.isBlank()) {
            return value;
        }
        List<Long> integers = unknown.getField(fieldNumber).getVarintList();
        if (!integers.isEmpty()) {
            // 同一字段重复编码时取最后一个整数值。
            return Long.toString(integers.getLast());
        }
        throw new IllegalArgumentException("Missing request ID");
    }

}
