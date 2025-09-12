package org.flexlb.domain.batch;

import lombok.Getter;
import org.flexlb.domain.ImmutableIntList;
import org.flexlb.domain.balance.BalanceContext;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * @author zjw
 * description:
 * date: 2025/3/12
 */
public class BucketQueueItem<Req, Resp> {

    private final AtomicBoolean consumed = new AtomicBoolean(false);

    @Getter
    private CompletableFuture<Resp> lbResultFuture = new CompletableFuture<>();

    @Getter
    private BalanceContext balanceContext;

    @Getter
    private Req originRequest;

    @Getter
    private ImmutableIntList tokenIds;

    @Getter
    private int tokenNum;

    public BucketQueueItem(BalanceContext balanceContext, Req originRequest, ImmutableIntList tokenIds) {
        this.balanceContext = balanceContext;
        this.originRequest = originRequest;
        this.tokenIds = tokenIds;
        this.tokenNum = tokenIds.size();
    }

    public boolean tryMarkItemConsumed() {
        return consumed.compareAndSet(false, true);
    }

    public void forceMarkConsumed() {
        consumed.set(true);
    }
}
