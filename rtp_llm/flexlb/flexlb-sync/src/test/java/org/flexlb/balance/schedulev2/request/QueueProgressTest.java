package org.flexlb.balance.schedulev2.request;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class QueueProgressTest {

    @Test
    void rollbackAndDecodeReservationAccumulateOnlyTheirOwnQueueTime() {
        QueueProgress progress = new QueueProgress(Stage.MAIN_WAITING, 0L, 0L, 0L, 0L, Stage.MAIN_WAITING);
        progress = progress.next(Stage.PREFILL_WAITING, 10_000_000L);
        progress = progress.next(Stage.DECODE_WAITING, 30_000_000L);
        progress = progress.next(Stage.DECODE_RESERVED, 60_000_000L);
        progress = progress.next(Stage.MAIN_WAITING, 100_000_000L);
        progress = progress.next(Stage.PREFILL_WAITING, 110_000_000L);
        progress = progress.next(Stage.DECODE_WAITING, 120_000_000L);
        progress = progress.next(Stage.CANCELLED, 130_000_000L);
        assertEquals(20L, progress.mainMs());
        assertEquals(30L, progress.prefillMs());
        assertEquals(80L, progress.decodeMs());
        assertEquals(130L, progress.mainMs() + progress.prefillMs() + progress.decodeMs());
        assertEquals(Stage.DECODE_WAITING, progress.previousStage());
    }
}
