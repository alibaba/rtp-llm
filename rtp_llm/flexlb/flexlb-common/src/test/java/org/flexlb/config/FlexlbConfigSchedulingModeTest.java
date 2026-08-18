package org.flexlb.config;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

class FlexlbConfigSchedulingModeTest {

    @Test
    void batch_should_use_scheduler_and_master_batch_delivery_regardless_of_auto_tpm() {
        FlexlbConfig config = config("BATCH", false);

        assertTrue(config.usesPriorityScheduler());
        assertTrue(config.usesBatchEnqueueDelivery());
        assertFalse(config.usesRouteDecisionDelivery());

        config.setAutoTpmEnabled(true);

        assertTrue(config.usesPriorityScheduler());
        assertTrue(config.usesBatchEnqueueDelivery());
        assertFalse(config.usesRouteDecisionDelivery());
    }

    @Test
    void queue_with_auto_tpm_should_use_scheduler_and_route_decision_delivery() {
        FlexlbConfig config = config("QUEUE", true);

        assertTrue(config.usesPriorityScheduler());
        assertTrue(config.usesRouteDecisionDelivery());
        assertFalse(config.usesBatchEnqueueDelivery());
    }

    @Test
    void legacy_queue_should_bypass_scheduler() {
        FlexlbConfig config = config("QUEUE", false);

        assertFalse(config.usesPriorityScheduler());
        assertFalse(config.usesRouteDecisionDelivery());
        assertFalse(config.usesBatchEnqueueDelivery());
    }

    @Test
    void direct_should_bypass_scheduler_even_when_auto_tpm_is_enabled() {
        FlexlbConfig config = config("DIRECT", true);

        assertFalse(config.usesPriorityScheduler());
        assertFalse(config.usesRouteDecisionDelivery());
        assertFalse(config.usesBatchEnqueueDelivery());
    }

    private static FlexlbConfig config(String mode, boolean autoTpmEnabled) {
        FlexlbConfig config = new FlexlbConfig();
        config.setDefaultScheduleMode(mode);
        config.setAutoTpmEnabled(autoTpmEnabled);
        return config;
    }
}
