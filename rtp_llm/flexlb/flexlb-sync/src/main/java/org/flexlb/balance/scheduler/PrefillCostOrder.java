package org.flexlb.balance.scheduler;

import java.math.BigInteger;
import java.util.Comparator;
import java.util.List;

/** Input-token benefit per unit of new Prefill computation, within business priority. */
final class PrefillCostOrder {
    record Candidate<T>(T value, int priority, long sequence, long inputTokens, Long computeTokens) { }

    static <T> List<T> first(List<Candidate<T>> ready, boolean priorityOrdering, int limit) {
        if (limit <= 0 || ready.isEmpty()) { return List.of(); }
        return ready.stream().sorted(comparator(priorityOrdering))
                .limit(limit).map(Candidate::value).toList();
    }

    static <T> Comparator<Candidate<T>> comparator(boolean priorityOrdering) {
        return Comparator.<Candidate<T>>comparingInt(candidate -> priorityOrdering ? -candidate.priority() : 0)
                .thenComparing(PrefillCostOrder::compareEstimates)
                .thenComparingLong(Candidate::sequence);
    }

    private static int compareEstimates(Candidate<?> left, Candidate<?> right) {
        boolean leftUnknown = unknownCost(left);
        boolean rightUnknown = unknownCost(right);
        if (leftUnknown || rightUnknown) { return Boolean.compare(leftUnknown, rightUnknown); }
        int ratio = compareUnitCost(left, right);
        if (ratio != 0) { return ratio; }
        int compute = Long.compare(left.computeTokens(), right.computeTokens());
        return compute != 0 ? compute : Long.compare(left.inputTokens(), right.inputTokens());
    }

    private static boolean unknownCost(Candidate<?> candidate) {
        return candidate.inputTokens() <= 0 || candidate.computeTokens() == null
                || candidate.computeTokens() < 0 || candidate.computeTokens() > candidate.inputTokens();
    }

    /** Lower C/T means more successful input tokens (including cache) per compute token.
     * For a common cache-read weight 0 <= a < 1, (C + a*H)/T has the same order.
     * Exact cross multiplication avoids floating-point ties and long overflow.
     */
    private static int compareUnitCost(Candidate<?> a, Candidate<?> b) {
        try {
            return Long.compare(Math.multiplyExact(a.computeTokens(), b.inputTokens()),
                    Math.multiplyExact(b.computeTokens(), a.inputTokens()));
        } catch (ArithmeticException overflow) {
            return BigInteger.valueOf(a.computeTokens()).multiply(BigInteger.valueOf(b.inputTokens()))
                    .compareTo(BigInteger.valueOf(b.computeTokens()).multiply(BigInteger.valueOf(a.inputTokens())));
        }
    }
}
