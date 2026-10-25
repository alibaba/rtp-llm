package org.flexlb.balance.schedulev2;

import org.flexlb.balance.endpoint.DecodeEndpoint;

/**
 * 出队检查的固定原因标签，不把请求 ID、异常文本或动态配置值放进指标。
 */
public enum QueueBlockReason {
    NO_AVAILABLE_PREFILL, SHARED_QUEUE_FULL, CANDIDATES_CHANGED, PREFILL_INFLIGHT_FULL,
    PREFILL_KV_FULL,
    NO_AVAILABLE_DECODE, REQUEST_EXCEEDS_KV_BUDGET, DECODE_REQUESTS_FULL, DECODE_KV_FULL, DECODE_REQUESTS_AND_KV_FULL,
    RESERVATION_CHANGED, ENDPOINT_RETIRED;

    /**
     * 同时缺少请求名额和 KV 时使用组合原因，不丢掉其中一个容量约束。
     */
    public static QueueBlockReason decodeCapacity(DecodeEndpoint.CapacityDeficit deficit) {
        if (deficit.requests() > 0L) {
            return deficit.needsKv() ? DECODE_REQUESTS_AND_KV_FULL : DECODE_REQUESTS_FULL;
        }
        return deficit.needsKv() ? DECODE_KV_FULL : RESERVATION_CHANGED;
    }
}
