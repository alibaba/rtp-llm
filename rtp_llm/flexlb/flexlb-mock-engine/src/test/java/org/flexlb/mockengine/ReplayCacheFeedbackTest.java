package org.flexlb.mockengine;

import java.util.List;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class ReplayCacheFeedbackTest {
    @Test
    void delayedReplyKeepsItsCapturedKeysAndANoopPollHasNoIndexDelta() throws Exception {
        try (var harness = new AutoTpmE2EHarness(58440, 1, 1, "10", 1, true)) {
            var worker = new ReplayPrefillWorker(harness.prefillEngines.getFirst());
            assertFalse(worker.captureDelta().changed());
            MockEngineTestSupport.seedCache(worker.cache, List.of(1L));
            var first = worker.captureDelta();

            worker.cache.evict(List.of(1L));
            MockEngineTestSupport.seedCache(worker.cache, List.of(2L));
            worker.cache.evict(List.of(2L));
            var second = worker.captureDelta();

            assertArrayEquals(new Long[]{1L}, first.added());
            assertArrayEquals(new Long[0], first.removed());
            assertArrayEquals(new Long[0], second.added());
            assertArrayEquals(new Long[]{1L}, second.removed());
            assertFalse(worker.captureDelta().changed());

            MockEngineTestSupport.seedCache(worker.cache, List.of(3L));
            worker.cache.evict(List.of(3L));
            var transientChange = worker.captureDelta();
            assertTrue(transientChange.touched());
            assertFalse(transientChange.changed());
        }
    }
}
