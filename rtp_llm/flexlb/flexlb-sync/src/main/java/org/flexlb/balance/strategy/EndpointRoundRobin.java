package org.flexlb.balance.strategy;

import org.flexlb.dao.route.RoleType;

import java.util.Arrays;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.IntFunction;
import java.util.function.IntPredicate;

/** Continues an address ring within each routing group, even as eligibility changes. */
final class EndpointRoundRobin {
    private record Scope(RoleType role, String group) { }
    private static final class Cursor { private String lastSelectedAddress; }
    private static final int MAX_CACHED_SIZE = 4096;
    private static final ThreadLocal<Scratch> SCRATCH = ThreadLocal.withInitial(() -> new Scratch(0));
    private static final class Scratch {
        private String[] eligibleAddresses;
        private String[] orderedAddresses;
        private boolean inUse;

        private Scratch(int size) { grow(size); }
        private void grow(int size) {
            var eligible = new String[size];
            var ordered = new String[size];
            eligibleAddresses = eligible;
            orderedAddresses = ordered;
        }
    }
    private final ConcurrentHashMap<Scope, Cursor> cursors = new ConcurrentHashMap<>();

    int next(RoleType role, String group, int size, IntPredicate eligible, IntFunction<String> address) {
        Cursor cursor = cursors.computeIfAbsent(new Scope(role, group), ignored -> new Cursor());
        Scratch cached = SCRATCH.get();
        Scratch scratch = cached;
        if (scratch.inUse || size < 0 || size > MAX_CACHED_SIZE) { scratch = new Scratch(size); }
        else if (scratch.eligibleAddresses.length < size) { scratch.grow(size); }
        scratch.inUse = true;
        try {
            // Eligibility belongs to this routing snapshot. Capture it before the shared cursor lock.
            String[] eligibleAddresses = scratch.eligibleAddresses;
            String[] orderedAddresses = scratch.orderedAddresses;
            int count = 0;
            boolean ordered = true;
            for (int i = 0; i < size; i++) {
                if (eligible.test(i)) {
                    String candidate = address.apply(i);
                    eligibleAddresses[i] = candidate;
                    if (ordered && count > 0 && (candidate == null || orderedAddresses[count - 1] == null
                            || orderedAddresses[count - 1].compareTo(candidate) > 0)) { ordered = false; }
                    orderedAddresses[count++] = candidate;
                }
            }
            if (count == 0) { return -1; }
            // Published directories are ordered; filtering preserves that order.
            if (!ordered) { Arrays.sort(orderedAddresses, 0, count); }
            String selected;
            synchronized (cursor) {
                int index = cursor.lastSelectedAddress == null ? 0
                        : Arrays.binarySearch(orderedAddresses, 0, count, cursor.lastSelectedAddress);
                if (cursor.lastSelectedAddress != null) {
                    if (index < 0) { index = -index - 1; }
                    else {
                        // Advance past the address, including duplicate entries in a captured snapshot.
                        while (index < count && orderedAddresses[index].equals(cursor.lastSelectedAddress)) { index++; }
                    }
                }
                selected = orderedAddresses[index == count ? 0 : index];
                cursor.lastSelectedAddress = selected;
            }
            // Keep the original index so callers can recover their exact captured endpoint.
            return Arrays.asList(eligibleAddresses).indexOf(selected);
        } finally {
            // Temporary captures have no retained owner; only the thread cache must release references.
            if (scratch == cached) {
                Arrays.fill(scratch.eligibleAddresses, 0, size, null);
                Arrays.fill(scratch.orderedAddresses, 0, size, null);
            }
            scratch.inUse = false;
        }
    }
}
