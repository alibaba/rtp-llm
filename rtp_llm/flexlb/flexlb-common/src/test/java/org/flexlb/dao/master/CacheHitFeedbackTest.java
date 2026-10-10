package org.flexlb.dao.master;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertThrows;

class CacheHitFeedbackTest {

    @Test
    void rejectsMissingWorkerIdentityAtConstruction() {
        assertThrows(NullPointerException.class, () -> new CacheHitFeedback(
                "finished", "request-1", "KVCM", "PREFILL", "default", null,
                "completed", 100, 64, 50, false, 0, 0, 40, -10));
    }
}
