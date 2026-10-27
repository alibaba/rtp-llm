package org.flexlb.mock.compute;

import static org.junit.jupiter.api.Assertions.*;
import java.time.Duration;
import java.util.*;
import java.util.concurrent.*;
import org.flexlb.mock.api.model.ComputeSpec.Noise;
import org.junit.jupiter.api.Test;

class NoiseModelTest {
    @Test void defaultsAreIdentityAndScaleIsAppliedOnce() {
        var plain = new NoiseModel(new Noise(null, null, null, null, null));
        assertEquals(Duration.ofNanos(125), plain.apply(Duration.ofNanos(125), "batch").planned());
        var scaled = new NoiseModel(new Noise(2.0, null, null, null, null));
        assertEquals(Duration.ofMillis(20), scaled.apply(Duration.ofMillis(10), "batch").planned());
    }

    @Test void seedAndInvocationIdentityAreStableAcrossOrderAndConcurrency() throws Exception {
        var noise = new NoiseModel(new Noise(1.0, 2.0, 0.02, 10.0, 42L));
        var expected = new HashMap<String, NoiseModel.Sample>();
        for (int i = 0; i < 100; i++) expected.put("engine/" + i, noise.apply(Duration.ofMillis(100), "engine/" + i));
        try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
            var keys = new ArrayList<>(expected.keySet()); Collections.shuffle(keys, new Random(17));
            var tasks = keys.stream().map(key -> executor.submit(() -> noise.apply(Duration.ofMillis(100), key))).toList();
            for (int i = 0; i < keys.size(); i++) assertEquals(expected.get(keys.get(i)), tasks.get(i).get());
        }
        assertNotEquals(expected.get("engine/1"), expected.get("engine/2"));
        assertNotEquals(expected.get("engine/1"), new NoiseModel(new Noise(1.0, 2.0, 0.02, 10.0, 43L))
            .apply(Duration.ofMillis(100), "engine/1"));
    }

    @Test void gaussianStaysBoundedAndDurationNeverBecomesNegative() {
        var noise = new NoiseModel(new Noise(1.0, 1000.0, 0.0, 5.0, 7L));
        for (int i = 0; i < 1000; i++) {
            var sample = noise.apply(Duration.ofMillis(1), "batch/" + i);
            assertTrue(Math.abs(sample.jitterMs()) <= 5);
            assertTrue(sample.planned().toNanos() >= 0 && sample.planned().toNanos() <= 6_000_000);
        }
    }

    @Test void rejectsInvalidNoiseAndOverflowInsteadOfSilentlyUsingAnotherModel() {
        assertThrows(IllegalArgumentException.class, () -> new Noise(1.0, 1.0, null, null, 1L));
        assertThrows(IllegalArgumentException.class, () -> new Noise(1.0, 1.0, null, 1.0, null));
        assertThrows(IllegalArgumentException.class, () -> new Noise(Double.NaN, null, null, null, null));
        assertThrows(IllegalArgumentException.class, () -> new Noise(-1.0, null, null, null, null));
        assertThrows(IllegalArgumentException.class, () -> new NoiseModel(new Noise(Double.MAX_VALUE, null, null, null, null))
            .apply(Duration.ofMillis(1), "batch"));
    }
}
