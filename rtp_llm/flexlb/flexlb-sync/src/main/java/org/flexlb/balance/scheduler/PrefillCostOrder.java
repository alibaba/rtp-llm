package org.flexlb.balance.scheduler;

import java.math.BigInteger;
import java.util.Comparator;
import java.util.function.ToIntFunction;
import java.util.function.ToLongFunction;

/** Input-token benefit per unit of new Prefill computation, within business priority. */
final class PrefillCostOrder {
    /** Worker placement freezes the selected endpoint's hit count for both dispatch and projection. */
    static <T> Comparator<T> workerOrder(ToIntFunction<T> priority, ToLongFunction<T> length,
                                         ToLongFunction<T> hits, Comparator<T> arrival) {
        return Comparator.comparingInt(priority).reversed().thenComparing((left, right) -> {
            long a = length.applyAsLong(left);
            long b = length.applyAsLong(right);
            if (a == 0 || b == 0) { return Boolean.compare(a == 0, b == 0); }
            return compareCosts(a, a - hits.applyAsLong(left), b, b - hits.applyAsLong(right));
        }).thenComparing(arrival);
    }

    private static int compareCosts(long lengthA, long computeA, long lengthB, long computeB) {
        int ratio = compareUnitCost(lengthA, computeA, lengthB, computeB);
        if (ratio != 0) { return ratio; }
        int compute = Long.compare(computeA, computeB);
        return compute != 0 ? compute : Long.compare(lengthA, lengthB);
    }

    /** Lower C/T means more successful input tokens (including cache) per compute token.
     * For a common cache-read weight 0 <= a < 1, (C + a*H)/T has the same order.
     * Exact cross multiplication avoids floating-point ties and long overflow.
     */
    private static int compareUnitCost(long lengthA, long computeA, long lengthB, long computeB) {
        try {
            return Long.compare(Math.multiplyExact(computeA, lengthB),
                    Math.multiplyExact(computeB, lengthA));
        } catch (ArithmeticException overflow) {
            return BigInteger.valueOf(computeA).multiply(BigInteger.valueOf(lengthB))
                    .compareTo(BigInteger.valueOf(computeB).multiply(BigInteger.valueOf(lengthA)));
        }
    }
}
