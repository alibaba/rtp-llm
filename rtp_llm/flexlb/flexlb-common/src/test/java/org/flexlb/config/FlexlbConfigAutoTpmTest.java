package org.flexlb.config;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;

/**
 * Tests for the Auto-TPM configuration surface of {@link FlexlbConfig}:
 * switch defaults and {@link FlexlbConfig#resolveSchedulingTargetMs(long, int)}.
 */
class FlexlbConfigAutoTpmTest {

    @Test
    void autoTpmSwitchesDefaultToFalse() {
        FlexlbConfig config = new FlexlbConfig();

        assertFalse(config.isAutoTpmPriorityQueueEnabled());
        assertFalse(config.isAutoTpmSchedulingTargetEnabled());
    }

    @Test
    void disabledSchedulingTargetDelegatesToResolveSloMs() {
        FlexlbConfig config = new FlexlbConfig();
        config.setCostSloMs(500);
        config.setCostSloBuckets("100:200,1000:400");

        for (int priority : new int[]{30, 50, 70}) {
            assertEquals(config.resolveSloMs(50), config.resolveSchedulingTargetMs(50, priority));
            assertEquals(config.resolveSloMs(500), config.resolveSchedulingTargetMs(500, priority));
            assertEquals(config.resolveSloMs(5_000), config.resolveSchedulingTargetMs(5_000, priority));
        }
    }

    @Test
    void enabledSchedulingTargetAppliesBucketTimesMultiplier() {
        FlexlbConfig config = new FlexlbConfig();
        config.setAutoTpmSchedulingTargetEnabled(true);
        // defaults: buckets 256:150,1024:300,4096:600,16384:1200,*:2400
        //           multipliers 30:2.0,40:1.5,50:1.0,60:0.75,70:0.5

        assertEquals(150, config.resolveSchedulingTargetMs(256, 50));
        assertEquals(300, config.resolveSchedulingTargetMs(1_000, 50));
        assertEquals(600, config.resolveSchedulingTargetMs(4_096, 50));
        assertEquals(1_200, config.resolveSchedulingTargetMs(16_384, 50));
        // wildcard bucket
        assertEquals(2_400, config.resolveSchedulingTargetMs(100_000, 50));

        // priority multipliers on the 1024 bucket
        assertEquals(600, config.resolveSchedulingTargetMs(1_000, 30));
        assertEquals(450, config.resolveSchedulingTargetMs(1_000, 40));
        assertEquals(225, config.resolveSchedulingTargetMs(1_000, 60));
        assertEquals(150, config.resolveSchedulingTargetMs(1_000, 70));
    }

    @Test
    void enabledSchedulingTargetUsesMultiplierOneForUnknownPriority() {
        FlexlbConfig config = new FlexlbConfig();
        config.setAutoTpmSchedulingTargetEnabled(true);

        assertEquals(300, config.resolveSchedulingTargetMs(1_000, 99));
    }

    @Test
    void settersInvalidateParsedCaches() {
        FlexlbConfig config = new FlexlbConfig();
        config.setAutoTpmSchedulingTargetEnabled(true);
        assertEquals(300, config.resolveSchedulingTargetMs(1_000, 50));

        config.setAutoTpmSloLengthBuckets("1024:1000,*:9000");
        config.setAutoTpmPrioritySloMultipliers("50:2.0");

        assertEquals(2_000, config.resolveSchedulingTargetMs(1_000, 50));
        assertEquals(9_000, config.resolveSchedulingTargetMs(100_000, 70));
    }
}
