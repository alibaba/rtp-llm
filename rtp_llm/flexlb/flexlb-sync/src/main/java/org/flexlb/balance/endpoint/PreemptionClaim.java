package org.flexlb.balance.endpoint;

import lombok.Getter;
import lombok.Setter;
import org.flexlb.balance.preemption.PreemptionCancelPhase;

/**
 * 保存一次抢占对请求容量的占用和取消进度；由 DecodeState 在节点锁内推进和释放。
 */
@Getter
@Setter
public final class PreemptionClaim {
    final long attemptToken;
    ClaimOwner owner;
    final long requiredKvTokens;
    final long kvBudgetTokens;
    PreemptionCancelPhase phase = PreemptionCancelPhase.CLAIMED;
    boolean returnedInstruction;
    boolean kvHeldAfterWorkerRelease;

    public PreemptionClaim(long attemptToken, ClaimOwner owner, long requiredKvTokens, long kvBudgetTokens) {
        if (requiredKvTokens < 0L || kvBudgetTokens < requiredKvTokens) {
            throw new IllegalArgumentException("Priority preemption claim requires expected KV >= hard KV >= 0");
        }
        this.attemptToken = attemptToken;
        this.owner = owner;
        this.requiredKvTokens = requiredKvTokens;
        this.kvBudgetTokens = kvBudgetTokens;
    }
}
