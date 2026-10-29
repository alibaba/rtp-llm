package org.flexlb.engine.grpc;

import com.google.protobuf.ByteString;
import org.flexlb.schedule.grpc.FlexlbScheduleProtocol;

/**
 * 在 Protobuf 边界统一读取请求 ID，业务层始终使用字符串，不生成或映射新的 ID。
 */
public final class RequestId {
    private RequestId() {}

    /**
     * 读取 Worker Status 的请求 ID，优先使用字符串字段。
     */
    public static String parse(EngineRpcService.TaskInfoPBOrBuilder message) {
        return preferString(message.getRequestIdStr(), message.getRequestId());
    }

    /**
     * 读取调度请求 ID，优先使用字符串字段。
     */
    public static String parse(FlexlbScheduleProtocol.FlexlbScheduleRequestPBOrBuilder message) {
        return preferString(message.getRequestIdStr(), message.getRequestId());
    }

    /**
     * 读取取消请求 ID，优先使用字符串字段。
     */
    public static String parse(FlexlbScheduleProtocol.FlexlbCancelRequestPBOrBuilder message) {
        return preferString(message.getRequestIdStr(), message.getRequestId());
    }

    /**
     * 读取状态查询 ID，优先使用字符串字段。
     */
    public static String parse(FlexlbScheduleProtocol.GetRequestStateRequestPBOrBuilder message) {
        return preferString(message.getRequestIdStr(), message.getRequestId());
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
        if (!readGenerateInputRequestId(generateInput).equals(requestId)) {
            throw new IllegalArgumentException(
                    "BATCH request_id must be the canonical int64 decimal string matching generate_input.request_id");
        }
    }

    /**
     * Read the generation identity without allocating the token or multimodal payload.
     * The delivery worker remains responsible for parsing nested generation inputs.
     */
    public static String readGenerateInputRequestId(ByteString bytes) {
        var input = bytes.newCodedInput();
        long requestId = 0L;
        try {
            int tag;
            while ((tag = input.readTag()) != 0) {
                if (tag == 8) {
                    requestId = input.readInt64();
                } else if (!input.skipField(tag)) {
                    throw new IllegalArgumentException("unexpected end-group in generate_input");
                }
            }
            input.checkLastTagWas(0);
        } catch (java.io.IOException malformed) {
            throw new IllegalArgumentException(
                    "generate_input must contain a valid GenerateInputPB", malformed);
        }
        return Long.toString(requestId);
    }

    /**
     * Encode an Engine cancellation ID without allowing a different spelling to cancel another Master ID.
     */
    public static long toEngineRequestId(String requestId) {
        long engineRequestId = Long.parseLong(requestId);
        if (!Long.toString(engineRequestId).equals(requestId)) {
            throw new IllegalArgumentException(
                    "Engine request_id must be a canonical int64 decimal string");
        }
        return engineRequestId;
    }

    private static String preferString(String stringRequestId, long numericId) {
        if (!stringRequestId.isBlank()) {
            return stringRequestId;
        }
        return Long.toString(numericId);
    }
}
