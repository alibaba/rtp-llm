package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class MockPrefillBatchPolicyTest {
    private MockPrefillBatchPolicy policy(long kv, int cp) {
        return new MockPrefillBatchPolicy(4, 100, kv, 1000, cp, false, 0, 20, 0);
    }

    @Test void checksCandidateBeforeAdmissionAndCanSkipIt() {
        var budget = policy(1000, 1).newBudget();
        assertTrue(budget.fits(60, 0)); budget.add(60, 0);
        assertFalse(budget.fits(40, 0), "equality is rejected, not admitted then stopped");
        assertTrue(budget.fits(39, 0), "a later smaller request can fit");
    }

    @Test void cacheAndComputeAreIndependentBudgets() {
        var budget = policy(500, 1).newBudget();
        budget.add(300, 280);
        assertTrue(budget.fits(190, 120));
        assertFalse(budget.fits(200, 190), "full KV sum reaches limit");
        assertFalse(budget.fits(90, 10), "compute sum reaches limit");
    }

    @Test void padsEachComputeSequenceForCp() {
        var budget = policy(1000, 8).newBudget();
        budget.add(49, 0); // 64 padded tokens
        assertTrue(budget.fits(32, 0));
        assertFalse(budget.fits(33, 0)); // 48 padded tokens
    }

    @Test void singletonMayExceedBatchBudgetButNotStandaloneBoundary() {
        var budget = policy(500, 1).newBudget();
        assertTrue(budget.fits(600, 0)); budget.add(600, 0);
        assertFalse(budget.fits(1, 0));
        assertFalse(policy(500, 1).newBudget().fits(1000, 0));
    }

    @Test void rectangleGateWhenNoSeparateKvBudget() {
        var budget = policy(0, 1).newBudget();
        budget.add(60, 55);
        assertFalse(budget.fits(10, 9), "padded 2x60 exceeds 100 despite sum=70");
    }

    @Test void countCapIsInclusiveAndStopQuotaIsDistinct() {
        var budget = policy(1000, 1).newBudget();
        for (int i = 0; i < 4; i++) {
            assertTrue(budget.fits(1, 0)); budget.add(1, 0);
        }
        assertFalse(budget.fits(1, 0));
        var stop = new MockPrefillBatchPolicy(4, 100, 1000, 1000, 1, false, 30, 20, 0).newBudget();
        stop.add(20, 0);
        assertTrue(stop.fits(20, 0)); stop.add(20, 0);
        assertFalse(stop.fits(1, 0));
    }

    @Test void forceSingleAndInvalidConfig() throws Exception {
        var budget = new MockPrefillBatchPolicy(4, 100, 1000, 1000, 1, true, 0, 20, 0).newBudget();
        budget.add(1, 0); assertFalse(budget.fits(1, 0));
        var node = new ObjectMapper().readTree("{}");
        assertThrows(IllegalArgumentException.class, () -> MockPrefillBatchPolicy.load(node));
    }
}
