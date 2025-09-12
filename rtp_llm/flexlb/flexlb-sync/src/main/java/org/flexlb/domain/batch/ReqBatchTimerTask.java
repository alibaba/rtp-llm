package org.flexlb.domain.batch;

import lombok.Getter;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.Delayed;
import java.util.concurrent.TimeUnit;

/**
 * @author zjw
 * description:
 * date: 2025/3/13
 */
@Getter
public class ReqBatchTimerTask implements Delayed {

    private String requestId;

    private CompletableFuture<Boolean> timerFuture;

    private long endTime;

    public ReqBatchTimerTask(String requestId, CompletableFuture<Boolean> timerFuture, long endTime) {
        this.requestId = requestId;
        this.timerFuture = timerFuture;
        this.endTime = endTime;
    }

    @Override
    public long getDelay(TimeUnit unit) {
        return unit.convert(endTime - System.currentTimeMillis(), TimeUnit.MILLISECONDS);
    }

    @Override
    public int compareTo(Delayed o) {
        return Long.compare(endTime - System.currentTimeMillis(), o.getDelay(TimeUnit.MILLISECONDS));
    }

}
