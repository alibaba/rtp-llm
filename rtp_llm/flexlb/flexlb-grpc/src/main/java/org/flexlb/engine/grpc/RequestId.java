package org.flexlb.engine.grpc;

import com.google.protobuf.ByteString;
import com.google.protobuf.InvalidProtocolBufferException;
import org.flexlb.schedule.grpc.FlexlbScheduleProtocol;

import java.util.OptionalLong;

/**
 * 在 Protobuf 边界优先读取 request_id_str，否则将 int64 request_id 转为字符串。
 * 业务层保留解析后的 ID，不生成或映射新的 ID。
 */
public final class RequestId {
    private RequestId() {
    }

    /**
     * 读取 Worker Status 的字符串或整数 ID。
     */
    public static String parse(EngineRpcService.TaskInfoPBOrBuilder message) {
        return parseString(message.getRequestIdStr(), message.getRequestId());
    }

    /**
     * 优先读取字符串 ID，否则读取整数 ID。OrBuilder 同时支持消息和 Builder。
     */
    public static String parse(FlexlbScheduleProtocol.FlexlbScheduleRequestPBOrBuilder message) {
        return parseString(message.getRequestIdStr(), message.getRequestId());
    }

    /**
     * 优先读取字符串 ID，否则读取整数 ID。OrBuilder 同时支持消息和 Builder。
     */
    public static String parse(FlexlbScheduleProtocol.FlexlbCancelRequestPBOrBuilder message) {
        return parseString(message.getRequestIdStr(), message.getRequestId());
    }

    /**
     * 优先读取字符串 ID，否则读取整数 ID。OrBuilder 同时支持消息和 Builder。
     */
    public static String parse(FlexlbScheduleProtocol.GetRequestStateRequestPBOrBuilder message) {
        return parseString(message.getRequestIdStr(), message.getRequestId());
    }

    /**
     * 将引擎消息中的 int64 请求 ID 转为字符串。
     */
    public static String parse(EngineRpcService.GenerateInputPBOrBuilder message) {
        return Long.toString(message.getRequestId());
    }

    /**
     * 将引擎消息中的 int64 请求 ID 转为字符串。
     */
    public static String parse(EngineRpcService.EnqueueBatchSuccessPBOrBuilder message) {
        return Long.toString(message.getRequestId());
    }

    /**
     * 将引擎消息中的 int64 请求 ID 转为字符串。
     */
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

    /**
     * 字符串 ID 写入独立字段；规范 int64 ID 同时写入整数字段供整数协议客户端读取。
     */
    public static void set(FlexlbScheduleProtocol.FlexlbCancelRequestPB.Builder message, String requestId) {
        message.setRequestIdStr(requestId);
        canonicalInteger(requestId).ifPresent(message::setRequestId);
    }

    /**
     * 生命周期响应保留字符串 ID，并为规范 int64 ID 填充整数协议字段。
     */
    public static void set(FlexlbScheduleProtocol.RequestLifecyclePB.Builder message, String requestId) {
        message.setRequestIdStr(requestId);
        canonicalInteger(requestId).ifPresent(message::setRequestId);
    }

    private static String parseString(String stringId, long integerId) {
        if (stringId.isEmpty()) {
            return Long.toString(integerId);
        }
        if (stringId.isBlank()) {
            throw new IllegalArgumentException("String request ID must not be blank");
        }
        return stringId;
    }

    private static OptionalLong canonicalInteger(String requestId) {
        try {
            long integerId = Long.parseLong(requestId);
            return Long.toString(integerId).equals(requestId) ? OptionalLong.of(integerId) : OptionalLong.empty();
        } catch (NumberFormatException error) {
            return OptionalLong.empty();
        }
    }
}
