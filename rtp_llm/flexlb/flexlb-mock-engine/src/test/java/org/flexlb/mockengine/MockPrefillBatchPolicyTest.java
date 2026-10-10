package org.flexlb.mockengine;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class MockPrefillBatchPolicyTest {
    private MockPrefillBatchPolicy policy(long kv, int cp) {
        return new MockPrefillBatchPolicy(4, 100, kv, 1000, cp, false, 0, 20, 0, false, cp > 1);
    }

    @Test void checksCandidateBeforeAdmissionAndCanSkipIt() {
        var budget = policy(1000, 1).newBudget();
        assertTrue(budget.fits(40, 0, 1)); budget.add(40, 0, 1);
        assertFalse(budget.fits(60, 0, 1), "equality is rejected, not admitted then stopped");
        assertTrue(budget.fits(39, 0, 1), "a later smaller request can fit");
    }

    @Test void cacheHitsDoNotRelaxFullTokenAndRectangleBudgets() {
        for (long kv : new long[]{0, 500, 1000}) {
            var budget = policy(kv, 1).newBudget();
            budget.add(60, 55, 1);
            assertFalse(budget.fits(10, 9, 1), "2x60 exceeds 100 despite cache hits");
            assertFalse(budget.fits(40, 39, 1), "full sum equals token limit");
        }
    }

    @Test void padsEachComputeSequenceForCpAndStopsOnNextCandidate() {
        var budget = new MockPrefillBatchPolicy(4, 1000, 0, 2000, 8, false, 96, 0, 0, false, true).newBudget();
        budget.add(49, 0, 1); // 64 padded tokens
        assertTrue(budget.fits(33, 0, 1)); // stop quota applies to the next admission
        budget.add(33, 0, 1); // 48 padded tokens => total 112
        assertFalse(budget.fits(1, 0, 1));
        var wide = new MockPrefillBatchPolicy(4, 1000, 0, 2000, 8, false, 96, 0, 0, false, true).newBudget();
        wide.add(49, 0, 2); // padding then width => 128
        assertFalse(wide.fits(1, 0, 1));
    }

    @Test void singletonMayExceedBatchBudgetButNotStandaloneBoundary() {
        var budget = policy(500, 1).newBudget();
        assertTrue(budget.fits(600, 0, 1)); budget.add(600, 0, 1);
        assertFalse(budget.fits(1, 0, 1));
        assertFalse(policy(500, 1).newBudget().fits(1000, 0, 1));
    }

    @Test void rectangleGateWhenNoSeparateKvBudget() {
        var budget = policy(0, 1).newBudget();
        budget.add(60, 55, 1);
        assertFalse(budget.fits(10, 9, 1), "padded 2x60 exceeds 100 despite sum=70");
    }

    @Test void countCapIsInclusiveAndStopQuotaIsDistinct() {
        var budget = policy(1000, 1).newBudget();
        for (int i = 0; i < 4; i++) {
            assertTrue(budget.fits(1, 0, 1)); budget.add(1, 0, 1);
        }
        assertFalse(budget.fits(1, 0, 1));
        var stop = new MockPrefillBatchPolicy(4, 100, 1000, 1000, 1, false, 30, 20, 0, false, false).newBudget();
        stop.add(20, 0, 1);
        assertTrue(stop.fits(20, 0, 1)); stop.add(20, 0, 1);
        assertFalse(stop.fits(1, 0, 1));
    }

    @Test void forceSingleAndInvalidConfig() throws Exception {
        var budget = new MockPrefillBatchPolicy(4, 100, 1000, 1000, 4, true, 0, 20, 0, false, true).newBudget();
        budget.add(1, 0, 1); assertFalse(budget.fits(1, 0, 1));
        var node = new ObjectMapper().readTree("{}");
        assertThrows(IllegalArgumentException.class, () -> MockPrefillBatchPolicy.load(node));
    }

    @Test void sequenceWidthAffectsTokenAndRectangleCosts() {
        var budget = policy(1000, 1).newBudget();
        budget.add(20, 19, 3);
        assertFalse(budget.fits(20, 19, 2));
        assertTrue(budget.fits(20, 19, 1));
        var rectangle = policy(1000, 1).newBudget();
        rectangle.add(30, 29, 2);
        assertFalse(rectangle.fits(1, 0, 2), "rectangle rejects sum=62");
        assertTrue(rectangle.fits(1, 0, 1));
    }

    @Test void archivedKvLimitRequiresExplicitFaultOptIn() {
        var normal = policy(10, 1).newBudget();
        normal.add(20, 0, 1);
        assertTrue(normal.fits(20, 0, 1));
        var fault = new MockPrefillBatchPolicy(4, 100, 30, 1000, 1, false, 0, 0, 0, true, false).newBudget();
        fault.add(20, 0, 1);
        assertFalse(fault.fits(20, 0, 1));
    }

    @Test void forceSingleDoesNotApplyWithoutCp() {
        var budget = new MockPrefillBatchPolicy(4, 100, 0, 1000, 1, true, 0, 0, 0, false, false).newBudget();
        budget.add(1, 0, 1);
        assertTrue(budget.fits(1, 0, 1));
        var enabledWithOneRank = new MockPrefillBatchPolicy(4, 100, 0, 1000, 1, true, 0, 0, 0, false, true).newBudget();
        enabledWithOneRank.add(1, 0, 1);
        assertFalse(enabledWithOneRank.fits(1, 0, 1));
    }

    @Test void initializedKvCanAdvanceAtCap() {
        var policy = new MockPrefillBatchPolicy(4, 100, 0, 1000, 1, false, 0, 0, 2, false, false);
        assertFalse(policy.allowsKvInitialization(2, false));
        assertTrue(policy.allowsKvInitialization(2, true));
        assertTrue(policy.allowsKvInitialization(1, false));
    }

    @Test void prefillWidthMatchesBeamAndReturnSequenceRules() {
        var input = MockEngineTestSupport.input(1, 20);
        var config = input.getGenerateConfig().toBuilder().setNumReturnSequences(4);
        var wide = input.toBuilder().setGenerateConfig(config).build();
        assertEquals(4, new MockPerformanceModel.RequestShape(wide, 20, 1, java.util.List.of(), 0, 0, false).prefillSequenceCount());
        var beam = input.toBuilder().setGenerateConfig(config.setNumBeams(4)).build();
        assertEquals(1, new MockPerformanceModel.RequestShape(beam, 20, 1, java.util.List.of(), 0, 0, false).prefillSequenceCount());
        var variable = input.toBuilder().setGenerateConfig(config.setNumBeams(1).addVariableNumBeams(1).addVariableNumBeams(8)).build();
        assertEquals(1, new MockPerformanceModel.RequestShape(variable, 20, 1, java.util.List.of(), 0, 0, false).prefillSequenceCount());
    }

    @Test void oversizedCandidatesDoNotOverflowAdmission() {
        var budget = policy(0, 1).newBudget();
        budget.add(1, 0, 1);
        assertFalse(budget.fits(Long.MAX_VALUE, 0, Integer.MAX_VALUE));
        assertThrows(IllegalArgumentException.class, () -> budget.fits(1, 0, 0));
        assertThrows(IllegalArgumentException.class, () -> budget.fits(1, 2, 1));
    }

    @Test void matchesRealFifoSchedulerRegressionVectors() {
        // FIFOSchedulerTest: prefillShapeLimitAppliesToOrdinaryAndGroupAdmission,
        // prefillShapeIncludesReusedPrefixLength, prefillShapeUsesCurrentBatchSizeAsWidth.
        for (long hit : new long[]{0, 59}) {
            var budget = policy(3145728, 1).newBudget();
            assertTrue(budget.fits(60, hit, 1)); budget.add(60, hit, 1);
            assertFalse(budget.fits(39, 0, 1));
        }
        var wide = policy(3145728, 1).newBudget();
        wide.add(40, 0, 1);
        assertFalse(wide.fits(10, 0, 3));
        // FIFOSchedulerTest: withoutCacheQuotaUsesSharedZigzagPaddingAndStopsTail.
        var cp = new MockPrefillBatchPolicy(100, 100, 0, 8192, 2, false, 10, 0, 0, false, true).newBudget();
        assertTrue(cp.fits(5, 0, 1)); cp.add(5, 0, 1); // padded 8
        assertTrue(cp.fits(1, 0, 1)); cp.add(1, 0, 1); // padded 4
        assertFalse(cp.fits(1, 0, 1));
    }

    @Test void loaderUsesRealCpDefaultAndPreservesExplicitCapturedValues() throws Exception {
        var node = new ObjectMapper().createObjectNode();
        node.put("max_requests", 4).put("max_batch_tokens", 100).put("max_seq_len", 1000);
        node.put("cp_size", 4);
        var implicit = MockPrefillBatchPolicy.load(node).newBudget();
        implicit.add(1, 0, 1);
        assertFalse(implicit.fits(1, 0, 1), "real CP force-single defaults to true");
        node.put("force_single", false);
        var captured = MockPrefillBatchPolicy.load(node).newBudget();
        captured.add(1, 0, 1);
        assertTrue(captured.fits(1, 0, 1), "explicit real capture must be preserved");
        node.put("cp_enabled", false);
        assertThrows(IllegalArgumentException.class, () -> MockPrefillBatchPolicy.load(node));
    }
}
