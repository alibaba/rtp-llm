package org.flexlb.balance.strategy;

import org.flexlb.dao.route.RoleType;

import java.util.concurrent.ConcurrentHashMap;

/**
 * Continues an address ring within each routing group, even as eligibility changes.
 */
final class EndpointRoundRobin {
    private record Scope(RoleType role, String group) {}

    private static final class Cursor {
        private String lastSelectedAddress;
    }

    private final ConcurrentHashMap<Scope, Cursor> cursors = new ConcurrentHashMap<>();

    /**
     * Candidates are caller-owned indexes into an address-ordered snapshot. The eligible
     * prefix preserves that order and both arrays remain unchanged throughout this call.
     */
    int next(RoleType role, String group, String[] addresses, int[] eligible, int count) {
        if (count == 0) {
            return -1;
        }
        Cursor cursor = cursors.computeIfAbsent(new Scope(role, group), ignored -> new Cursor());
        synchronized (cursor) {
            int low = 0;
            int high = count;
            if (cursor.lastSelectedAddress != null) {
                // Upper bound skips the previous address, including duplicate observations.
                while (low < high) {
                    int mid = (low + high) >>> 1;
                    if (addresses[eligible[mid]].compareTo(cursor.lastSelectedAddress) <= 0) {
                        low = mid + 1;
                    } else {
                        high = mid;
                    }
                }
            }
            int selected = eligible[low == count ? 0 : low];
            cursor.lastSelectedAddress = addresses[selected];
            return selected;
        }
    }
}
