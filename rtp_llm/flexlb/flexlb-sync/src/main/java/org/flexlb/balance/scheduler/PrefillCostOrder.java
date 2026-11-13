package org.flexlb.balance.scheduler;

import java.math.BigInteger;
import java.util.Comparator;
import java.util.List;

/** Input-token benefit per unit of new Prefill computation, within business priority. */
final class PrefillCostOrder {
    record Candidate<T>(T value, int priority, long sequence, long inputTokens, Long computeTokens) { }

    static <T> List<T> first(List<Candidate<T>> ready, List<Candidate<T>> running,
            boolean priorityOrdering, int limit) {
        if (ready.isEmpty() || limit <= 0) { return List.of(); }
        Comparator<Candidate<T>> priority = Comparator.comparingInt(
                candidate -> priorityOrdering ? -candidate.priority() : 0);
        if (ready.stream().anyMatch(PrefillCostOrder::unknownCost)
                || running.stream().anyMatch(PrefillCostOrder::unknownCost)) {
            return ready.stream().sorted(priority.thenComparingLong(Candidate::sequence))
                    .limit(limit).map(Candidate::value).toList();
        }
        Comparator<Candidate<T>> order = priority.thenComparing(PrefillCostOrder::compareUnitCost)
                .thenComparingLong(Candidate::computeTokens)
                .thenComparingLong(Candidate::inputTokens).thenComparingLong(Candidate::sequence);
        Candidate<T> best = ready.stream().min(order).orElseThrow();
        for (var active : running) {
            if (priorityOrdering && active.priority() < best.priority()) { continue; }
            if ((priorityOrdering && active.priority() > best.priority())
                    || compareUnitCost(active, best) < 0) { return List.of(); }
        }
        // Equal benefit density stays parallel, including different request lengths.
        // Spare slots must not let low-benefit cold work race higher-benefit admission.
        return ready.stream().filter(candidate -> (!priorityOrdering || candidate.priority() == best.priority())
                        && compareUnitCost(candidate, best) == 0)
                .sorted(order).limit(limit).map(Candidate::value).toList();
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
