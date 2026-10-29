package org.flexlb.balance.strategy;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class PrefillComputeCapacityTest {
    @Test void warmExecutionCalibratesColdTokenBacklogWithoutColdPredictionSamples() {
        var capacity = new PrefillComputeCapacity(2000);
        assertEquals(0, capacity.estimatedMs(210000));
        capacity.observe(8190,117);
        assertEquals(3000,capacity.estimatedMs(210000));
        capacity.observe(0,1000);
        capacity.observe(1000,0);
        assertEquals(3000,capacity.estimatedMs(210000));
    }
    @Test void performanceChangesAdaptAndGenerationsStartWithoutStaleCapacity() {
        var capacity = new PrefillComputeCapacity(2000);
        capacity.observe(140000,2000);
        capacity.observe(126000,2000);
        assertEquals(1000,capacity.estimatedMs(63000));
        capacity.observe(154000,2000);
        assertEquals(1000,capacity.estimatedMs(77000));
        assertEquals(0,new PrefillComputeCapacity(2000).estimatedMs(77000));
    }
    @Test void oneTinySampleDoesNotReplaceTheWorkerCapacity() {
        var capacity = new PrefillComputeCapacity(2000);
        capacity.observe(140000,2000);
        capacity.observe(1,1);
        assertTrue(capacity.estimatedMs(70000)>=1000);
        assertTrue(capacity.estimatedMs(70000)<=1001);
    }
}
