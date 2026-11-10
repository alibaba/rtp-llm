package org.flexlb.mock.engine.forward;

import static org.junit.jupiter.api.Assertions.*;
import java.time.Duration;
import java.util.*;
import java.util.concurrent.*;
import org.junit.jupiter.api.Test;

/**
 * Checks reproducibility, bounds and defaults for timing noise.
 */
class NoiseModelTest {
    @Test void defaultsAreIdentity() {
        var plain = new NoiseModel(new NoiseModel.Config(null, null, null, null));
        assertEquals(Duration.ofNanos(125), plain.apply(Duration.ofNanos(125), "batch").planned());
    }

    @Test void seedAndInvocationIdentityAreStableAcrossOrderAndConcurrency() throws Exception {
        var noise = new NoiseModel(new NoiseModel.Config(2.0, 0.02, 10.0, 42L));
        var expected = new HashMap<String, NoiseModel.Sample>();
        for (int i = 0; i < 100; i++) expected.put("engine/" + i, noise.apply(Duration.ofMillis(100), "engine/" + i));
        try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
            var keys = new ArrayList<>(expected.keySet()); Collections.shuffle(keys, new Random(17));
            var tasks = keys.stream().map(key -> executor.submit(() -> noise.apply(Duration.ofMillis(100), key))).toList();
            for (int i = 0; i < keys.size(); i++) assertEquals(expected.get(keys.get(i)), tasks.get(i).get());
        }
        assertNotEquals(expected.get("engine/1"), expected.get("engine/2"));
        assertNotEquals(expected.get("engine/1"), new NoiseModel(new NoiseModel.Config(2.0, 0.02, 10.0, 43L))
            .apply(Duration.ofMillis(100), "engine/1"));
    }

    @Test void gaussianStaysBoundedAndDurationNeverBecomesNegative() {
        var noise = new NoiseModel(new NoiseModel.Config(1000.0, 0.0, 5.0, 7L));
        for (int i = 0; i < 1000; i++) {
            var sample = noise.apply(Duration.ofMillis(1), "batch/" + i);
            assertTrue(Math.abs(sample.jitterMs()) <= 5);
            assertTrue(sample.planned().toNanos() >= 0 && sample.planned().toNanos() <= 6_000_000);
        }
    }

    @Test void rejectsInvalidNoiseInsteadOfSilentlyUsingAnotherModel() {
        assertThrows(IllegalArgumentException.class, () -> new NoiseModel.Config(1.0, null, null, 1L));
        assertThrows(IllegalArgumentException.class, () -> new NoiseModel.Config(1.0, null, 1.0, null));
        assertThrows(IllegalArgumentException.class, () -> new NoiseModel.Config(Double.NaN, null, null, null));
        assertThrows(IllegalArgumentException.class, () -> new NoiseModel.Config(-1.0, null, null, null));
    }
}
