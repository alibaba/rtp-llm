package org.flexlb.mockengine;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class FaultInjectionConfigTest {

    @Test
    void defaultsAreSensible() {
        FaultInjectionConfig config = FaultInjectionConfig.builder().build();
        assertFalse(config.isFailOnEnqueue());
        assertEquals(13, config.getEnqueueErrorCode());
        assertEquals("mock enqueue failure", config.getEnqueueErrorMessage());
        assertEquals(0, config.getEnqueueDelayMs());
        assertFalse(config.isGenerateError());
        assertFalse(config.isFetchError());
        assertFalse(config.isNoRespond());
        assertEquals(0, config.getKvPressureTokens());
        assertEquals(0, config.getQueueDepthLimit());
        assertEquals(0, config.getCrashAfterNRequests());
    }

    @Test
    void builderSetsAllFields() {
        FaultInjectionConfig config = FaultInjectionConfig.builder()
                .failOnEnqueue(true)
                .enqueueErrorCode(42)
                .enqueueErrorMessage("custom error")
                .enqueueDelayMs(500)
                .generateError(true)
                .fetchError(true)
                .noRespond(true)
                .kvPressureTokens(999_999)
                .queueDepthLimit(20)
                .crashAfterNRequests(7)
                .build();

        assertTrue(config.isFailOnEnqueue());
        assertEquals(42, config.getEnqueueErrorCode());
        assertEquals("custom error", config.getEnqueueErrorMessage());
        assertEquals(500, config.getEnqueueDelayMs());
        assertTrue(config.isGenerateError());
        assertTrue(config.isFetchError());
        assertTrue(config.isNoRespond());
        assertEquals(999_999, config.getKvPressureTokens());
        assertEquals(20, config.getQueueDepthLimit());
        assertEquals(7, config.getCrashAfterNRequests());
    }

    @Test
    void toBuilderPreservesAllFields() {
        FaultInjectionConfig original = FaultInjectionConfig.builder()
                .failOnEnqueue(true)
                .enqueueDelayMs(200)
                .generateError(true)
                .kvPressureTokens(100_000)
                .crashAfterNRequests(3)
                .build();

        FaultInjectionConfig copy = original.toBuilder().build();

        assertEquals(original.isFailOnEnqueue(), copy.isFailOnEnqueue());
        assertEquals(original.getEnqueueErrorCode(), copy.getEnqueueErrorCode());
        assertEquals(original.getEnqueueErrorMessage(), copy.getEnqueueErrorMessage());
        assertEquals(original.getEnqueueDelayMs(), copy.getEnqueueDelayMs());
        assertEquals(original.isGenerateError(), copy.isGenerateError());
        assertEquals(original.isFetchError(), copy.isFetchError());
        assertEquals(original.isNoRespond(), copy.isNoRespond());
        assertEquals(original.getKvPressureTokens(), copy.getKvPressureTokens());
        assertEquals(original.getQueueDepthLimit(), copy.getQueueDepthLimit());
        assertEquals(original.getCrashAfterNRequests(), copy.getCrashAfterNRequests());
    }

    @Test
    void toBuilderAllowsPartialModification() {
        FaultInjectionConfig original = FaultInjectionConfig.builder()
                .failOnEnqueue(true)
                .enqueueDelayMs(200)
                .build();

        FaultInjectionConfig modified = original.toBuilder()
                .failOnEnqueue(false)
                .build();

        assertFalse(modified.isFailOnEnqueue());
        assertEquals(200, modified.getEnqueueDelayMs(), "unchanged fields should be preserved");
    }

    @Test
    void toStringContainsKeyFields() {
        FaultInjectionConfig config = FaultInjectionConfig.builder()
                .failOnEnqueue(true)
                .kvPressureTokens(500)
                .build();

        String str = config.toString();
        assertTrue(str.contains("failOnEnqueue=true"));
        assertTrue(str.contains("kvPressureTokens=500"));
    }
}
