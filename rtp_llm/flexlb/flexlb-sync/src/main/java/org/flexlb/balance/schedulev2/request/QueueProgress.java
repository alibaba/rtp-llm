package org.flexlb.balance.schedulev2.request;

import java.util.concurrent.TimeUnit;

/**
 * 阶段及累计排队时间随同一次 CAS 更新。退回主队列会累加新的一段，不会覆盖之前的等待。
 */
public record QueueProgress(Stage stage,
                            long enteredAtNanos,
                            long mainNanos,
                            long prefillNanos,
                            long decodeNanos,
                            Stage previousStage) {

    public QueueProgress next(Stage next, long nowNanos) {
        long elapsed = Math.max(0L, nowNanos - enteredAtNanos);
        return new QueueProgress(next, nowNanos, mainNanos + (stage == Stage.MAIN_WAITING ? elapsed : 0L),
                prefillNanos + (stage == Stage.PREFILL_WAITING ? elapsed : 0L),
                decodeNanos + (stage == Stage.DECODE_WAITING || stage == Stage.DECODE_RESERVED ? elapsed : 0L), stage);
    }

    public long mainMs() {
        return TimeUnit.NANOSECONDS.toMillis(mainNanos);
    }

    public long prefillMs() {
        return TimeUnit.NANOSECONDS.toMillis(prefillNanos);
    }

    public long decodeMs() {
        return TimeUnit.NANOSECONDS.toMillis(decodeNanos);
    }
}
