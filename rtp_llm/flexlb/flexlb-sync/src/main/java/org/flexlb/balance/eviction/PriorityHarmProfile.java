package org.flexlb.balance.eviction;

import org.flexlb.util.PriorityNormalizer;

import java.math.BigInteger;
import java.util.Arrays;

/**
 * Exact, overflow-free priority harm of an eviction plan.
 *
 * <p>Each valid priority (1-100) owns an independent bucket. Plans compare
 * buckets from priority 100 down to 1, so any positive harm at a higher
 * priority is worse than an arbitrary amount of harm at every lower
 * priority. The value in a bucket retains the existing case/stage/length
 * weighting; {@link BigInteger} prevents a large victim set from wrapping.
 */
public final class PriorityHarmProfile implements Comparable<PriorityHarmProfile> {

    private static final int MAX_PRIORITY = 100;
    private static final PriorityHarmProfile EMPTY =
            new PriorityHarmProfile(new BigInteger[MAX_PRIORITY + 1]);

    private final BigInteger[] harmByPriority;

    private PriorityHarmProfile(BigInteger[] harmByPriority) {
        this.harmByPriority = harmByPriority;
    }

    public static PriorityHarmProfile empty() {
        return EMPTY;
    }

    public static Builder builder() {
        return new Builder();
    }

    /** Add two profiles without losing precision. */
    public PriorityHarmProfile plus(PriorityHarmProfile other) {
        BigInteger[] sum = new BigInteger[MAX_PRIORITY + 1];
        for (int priority = 1; priority <= MAX_PRIORITY; priority++) {
            BigInteger left = harmByPriority[priority];
            BigInteger right = other.harmByPriority[priority];
            if (left != null || right != null) {
                sum[priority] = valueOrZero(left).add(valueOrZero(right));
            }
        }
        return new PriorityHarmProfile(sum);
    }

    /**
     * Smaller harm is preferable. The highest exact priority at which two
     * plans differ decides the result; lower-priority buckets are considered
     * only when every higher-priority bucket is equal.
     */
    @Override
    public int compareTo(PriorityHarmProfile other) {
        for (int priority = MAX_PRIORITY; priority >= 1; priority--) {
            int comparison = valueOrZero(harmByPriority[priority])
                    .compareTo(valueOrZero(other.harmByPriority[priority]));
            if (comparison != 0) {
                return comparison;
            }
        }
        return 0;
    }

    @Override
    public boolean equals(Object object) {
        return object instanceof PriorityHarmProfile other
                && Arrays.equals(harmByPriority, other.harmByPriority);
    }

    @Override
    public int hashCode() {
        return Arrays.hashCode(harmByPriority);
    }

    private static BigInteger valueOrZero(BigInteger value) {
        return value == null ? BigInteger.ZERO : value;
    }

    private static void requireValidPriority(int priority) {
        if (!PriorityNormalizer.isValid(priority)) {
            throw new IllegalArgumentException("priority must be in [1, 100]: " + priority);
        }
    }

    /** Mutable accumulator used only while one immutable plan is built. */
    public static final class Builder {

        private final BigInteger[] harmByPriority = new BigInteger[MAX_PRIORITY + 1];

        private Builder() {
        }

        public Builder add(int priority, long harm) {
            if (harm < 0) {
                throw new IllegalArgumentException("harm must be non-negative: " + harm);
            }
            return add(priority, BigInteger.valueOf(harm));
        }

        public Builder add(int priority, BigInteger harm) {
            requireValidPriority(priority);
            if (harm.signum() < 0) {
                throw new IllegalArgumentException("harm must be non-negative: " + harm);
            }
            if (harm.signum() != 0) {
                harmByPriority[priority] = valueOrZero(harmByPriority[priority]).add(harm);
            }
            return this;
        }

        public PriorityHarmProfile build() {
            return new PriorityHarmProfile(Arrays.copyOf(harmByPriority, harmByPriority.length));
        }
    }
}
