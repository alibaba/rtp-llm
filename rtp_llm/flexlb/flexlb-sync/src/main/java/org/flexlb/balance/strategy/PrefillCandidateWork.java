package org.flexlb.balance.strategy;

import org.flexlb.balance.endpoint.PrefillEndpoint;

/**
 * 请求在一台候选 Prefill 上的缓存命中和自身执行耗时，不包含排队等待。
 */
public record PrefillCandidateWork(String address,
                                   PrefillEndpoint endpoint,
                                   long cacheHitTokens,
                                   long routingCacheMatchTokens,
                                   long predictedMs) {
}
