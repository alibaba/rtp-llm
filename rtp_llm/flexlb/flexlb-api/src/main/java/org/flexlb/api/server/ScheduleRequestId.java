package org.flexlb.api.server;

import org.flexlb.api.protocol.FlexlbScheduleProtocol;

/**
 * Reads scheduling request IDs, preferring the string field over the legacy numeric field.
 */
public final class ScheduleRequestId {
    private ScheduleRequestId() {}

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

    private static String preferString(String stringRequestId, long numericId) {
        return stringRequestId.isBlank() ? Long.toString(numericId) : stringRequestId;
    }
}
