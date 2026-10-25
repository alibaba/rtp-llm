package org.flexlb.balance.schedulev2.request;

import org.flexlb.balance.endpoint.DecodeEndpoint;
import org.flexlb.balance.scheduler.decode.DecodeRequirement;
import org.flexlb.dao.loadbalance.ServerStatus;

/**
 * Decode 队列给请求选好并 reserve 的 Decode。
 *
 * @param endpoint     选中的 Decode
 * @param reservation  在这台 Decode 上的 reserve，拿 permit 和归还时都用它
 * @param serverStatus 返回给调用方的 Decode 地址等信息
 * @param requirement  reserve 时用的 KV 需求和容量上限，拿 permit 时用同一份
 */
public record DecodeChoice(DecodeEndpoint endpoint,
                           DecodeEndpoint.ReservationHandle reservation,
                           ServerStatus serverStatus,
                           DecodeRequirement requirement) {
}
