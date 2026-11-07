package org.flexlb.balance.strategy;

import org.flexlb.dao.route.RoleType;
import org.junit.jupiter.api.Test;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Random;
import java.util.TreeSet;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

class EndpointRoundRobinTest {
    @Test
    void changingMembershipFollowsAddressSuccessorWithinEachRoleAndGroup() {
        var rotation = new EndpointRoundRobin();
        var last = new HashMap<String, String>();
        var random = new Random(187237);
        for (int iteration = 0; iteration < 2000; iteration++) {
            int size = random.nextInt(80);
            String group = "g" + random.nextInt(4);
            var role = random.nextBoolean() ? RoleType.PREFILL : RoleType.DECODE;
            String scope = role + group;
            String[] addresses = new String[size];
            int[] eligible = new int[size];
            for (int i = 0; i < size; i++) {
                addresses[i] = "worker-" + random.nextInt(100);
            }
            Arrays.sort(addresses);
            var ring = new TreeSet<String>();
            int count = 0;
            for (int i = 0; i < size; i++) {
                if (random.nextBoolean()) {
                    eligible[count++] = i;
                    ring.add(addresses[i]);
                }
            }
            String expected =
                    ring.isEmpty()
                            ? null
                            : last.containsKey(scope) ? ring.higher(last.get(scope)) : ring.first();
            if (expected == null && !ring.isEmpty()) {
                expected = ring.first();
            }
            int selected = rotation.next(role, group, addresses, eligible, count);
            if (expected == null) {
                assertEquals(-1, selected);
            } else {
                assertEquals(expected, addresses[selected]);
                assertTrue(Arrays.stream(eligible, 0, count).anyMatch(i -> i == selected));
                last.put(scope, expected);
            }
        }
    }

    @Test
    void emptySelectionDoesNotAdvanceAndUnusedArrayTailIsIgnored() {
        var rotation = new EndpointRoundRobin();
        String[] addresses = {"a", "b", "c", null};
        int[] indexes = {0, 1, 2, -1};
        assertEquals(0, rotation.next(RoleType.DECODE, "g", addresses, indexes, 3));
        assertEquals(-1, rotation.next(RoleType.DECODE, "g", addresses, indexes, 0));
        assertEquals(1, rotation.next(RoleType.DECODE, "g", addresses, indexes, 3));
        indexes[0] = 2;
        assertEquals(2, rotation.next(RoleType.DECODE, "g", addresses, indexes, 1));
        indexes[0] = 0;
        assertEquals(0, rotation.next(RoleType.DECODE, "g", addresses, indexes, 3));
    }

    @Test
    void concurrentSelectionsAdvanceExactlyOnceAcross750Workers() throws Exception {
        var rotation = new EndpointRoundRobin();
        String[] addresses = new String[750];
        int[] indexes = new int[750];
        for (int i = 0; i < addresses.length; i++) {
            addresses[i] = "worker-" + i;
            indexes[i] = i;
        }
        Arrays.sort(addresses);
        try (var executor = Executors.newFixedThreadPool(16)) {
            var calls = new ArrayList<Future<Integer>>();
            for (int i = 0; i < 6000; i++) {
                calls.add(
                        executor.submit(
                                () ->
                                        rotation.next(
                                                RoleType.DECODE,
                                                "fleet",
                                                addresses,
                                                indexes,
                                                indexes.length)));
            }
            int[] counts = new int[addresses.length];
            for (var call : calls) {
                counts[call.get(10, TimeUnit.SECONDS)]++;
            }
            for (int count : counts) {
                assertEquals(8, count);
            }
        }
    }
}
