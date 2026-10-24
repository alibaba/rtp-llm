package org.flexlb.mockengine;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.List;
import java.util.function.LongSupplier;

/** FIFOScheduler's waiting/loading lanes. The engine owns the scheduler lock. */
final class MockFifoPrefillQueue<T> {
    enum State { WAITING, LOADING_CACHE, RUNNING, FINISHED }
    private enum Lane { NONE, NORMAL, GROUP }
    record Admission<T>(T value, State state) {}
    interface Round<T> { Admission<T> admit(T value); }
    interface Driver<T> {
        boolean hasError(T value);
        State advance(T value, State state);
        default T refreshed(T value) { return value; }
        Round<T> round(boolean group, List<T> alreadyRunning);
    }
    private final class Node {
        T value;
        State state = State.WAITING;
        final long enqueuedAt = clock.getAsLong();
        Node(T value) { this.value = value; }
    }
    private final LongSupplier clock;
    private final ArrayDeque<Node> ordinary = new ArrayDeque<>();
    private final ArrayDeque<Node> loading = new ArrayDeque<>();
    private final ArrayDeque<ArrayDeque<Node>> groups = new ArrayDeque<>();
    private final ArrayDeque<ArrayDeque<Node>> loadingGroups = new ArrayDeque<>();
    private Lane running = Lane.NONE;
    private boolean preferGroup;
    private long groupFallbacks;
    private long oldestAgeUs;

    MockFifoPrefillQueue() { this(System::nanoTime); }
    MockFifoPrefillQueue(LongSupplier clock) { this.clock = clock; }

    void enqueue(List<T> members, boolean group, int maxBatch, int maxInited) {
        var nodes = new ArrayDeque<Node>();
        members.forEach(member -> nodes.add(new Node(member)));
        if (group && members.size() <= maxBatch && (maxInited == 0 || members.size() <= maxInited)) groups.addLast(nodes);
        else {
            ordinary.addAll(nodes);
            if (group) groupFallbacks++;
        }
    }

    void executionFinished() { running = Lane.NONE; }

    List<T> poll(Driver<T> driver) {
        oldestAgeUs = 0;
        List<T> ready = new ArrayList<>();
        if (!loadingGroups.isEmpty()) {
            var head = loadingGroups.getFirst();
            advance(head, driver, true, ready, false);
            if (head.isEmpty()) loadingGroups.removeFirst();
            else if (running == Lane.NONE && head.stream().allMatch(node -> node.state == State.RUNNING)) {
                head.forEach(node -> ready.add(node.value));
                loadingGroups.removeFirst();
                running = Lane.GROUP;
                preferGroup = false;
                return ready;
            }
        }
        if (running != Lane.NONE) {
            if (running == Lane.NORMAL) {
                advance(loading, driver, false, ready, true);
                if (loadingGroupReady() || !groups.isEmpty()) preferGroup = true;
                else admitOrdinary(driver, ready);
            }
            return ready;
        }
        advance(loading, driver, false, ready, true);
        boolean hasGroup = !groups.isEmpty();
        if (!ready.isEmpty()) {
            running = Lane.NORMAL;
            preferGroup = hasGroup;
            if (!hasGroup) admitOrdinary(driver, ready);
        } else if (hasGroup && (preferGroup || ordinary.isEmpty())) {
            admitGroup(driver, ready);
            if (!ready.isEmpty()) { running = Lane.GROUP; preferGroup = false; }
            else {
                admitOrdinary(driver, ready);
                if (!ready.isEmpty()) { running = Lane.NORMAL; preferGroup = true; }
            }
        } else {
            admitOrdinary(driver, ready);
            if (!ready.isEmpty()) { running = Lane.NORMAL; preferGroup = hasGroup; }
            else if (hasGroup) {
                admitGroup(driver, ready);
                if (!ready.isEmpty()) { running = Lane.GROUP; preferGroup = false; }
            }
        }
        return ready;
    }

    private boolean loadingGroupReady() {
        return !loadingGroups.isEmpty() && loadingGroups.getFirst().stream().allMatch(node -> node.state == State.RUNNING);
    }

    private void advance(ArrayDeque<Node> nodes, Driver<T> driver, boolean group, List<T> ready, boolean move) {
        for (var iterator = nodes.iterator(); iterator.hasNext();) {
            var node = iterator.next();
            node.state = driver.advance(node.value, node.state);
            if (group && node.state == State.WAITING) node.state = driver.advance(node.value, node.state);
            node.value = driver.refreshed(node.value);
            if (node.state == State.FINISHED) iterator.remove();
            else if (move && node.state == State.WAITING) { iterator.remove(); ordinary.addLast(node); }
            else if (move && node.state == State.RUNNING) { iterator.remove(); ready.add(node.value); }
        }
    }

    private void admitOrdinary(Driver<T> driver, List<T> ready) {
        if (!ordinary.isEmpty()) oldestAgeUs = Math.max(0, (clock.getAsLong() - ordinary.stream()
                .mapToLong(node -> node.enqueuedAt).min().orElse(clock.getAsLong())) / 1000);
        scan(ordinary, driver, driver.round(false, List.copyOf(ready)), ready, null);
    }

    private void admitGroup(Driver<T> driver, List<T> ready) {
        if (!loadingGroups.isEmpty() || groups.isEmpty()) return;
        var head = groups.getFirst();
        // Real evaluates errors in the head group even when no live candidate fits.
        head.removeIf(node -> driver.hasError(node.value)
                && driver.advance(node.value, node.state) == State.FINISHED);
        if (head.isEmpty()) { groups.removeFirst(); return; }
        var admitted = new ArrayDeque<Node>();
        scan(head, driver, driver.round(true, List.of()), ready, admitted);
        if (head.isEmpty()) groups.removeFirst();
        else if (!admitted.isEmpty()) groupFallbacks++;
        if (admitted.isEmpty()) return;
        if (admitted.stream().allMatch(node -> node.state == State.RUNNING)) admitted.forEach(node -> ready.add(node.value));
        else loadingGroups.addLast(admitted);
    }

    private void scan(ArrayDeque<Node> waiting, Driver<T> driver, Round<T> round,
                      List<T> ready, ArrayDeque<Node> group) {
        for (var iterator = waiting.iterator(); iterator.hasNext();) {
            var node = iterator.next();
            if (driver.hasError(node.value)) {
                if (driver.advance(node.value, node.state) == State.FINISHED) iterator.remove();
                continue;
            }
            var admitted = round.admit(node.value);
            if (admitted == null) continue;
            node.value = admitted.value(); node.state = admitted.state();
            if (node.state == State.WAITING) continue;
            iterator.remove();
            if (node.state == State.FINISHED) continue;
            if (group != null) group.addLast(node);
            else if (node.state == State.LOADING_CACHE) loading.addLast(node);
            else ready.add(node.value);
        }
    }

    int ordinarySize() { return ordinary.size(); }
    int groupCount() { return groups.size(); }
    int size() { return ordinary.size() + groups.stream().mapToInt(ArrayDeque::size).sum(); }
    int loadingSize() { return loading.size() + loadingGroups.stream().mapToInt(ArrayDeque::size).sum(); }
    boolean hasWork() { return size() + loadingSize() > 0; }
    long oldestAgeUs() { return oldestAgeUs; }
    long takeGroupFallbacks() { long value = groupFallbacks; groupFallbacks = 0; return value; }
    void clear() {
        ordinary.clear(); loading.clear(); groups.clear(); loadingGroups.clear();
        running = Lane.NONE; preferGroup = false; groupFallbacks = oldestAgeUs = 0;
    }
}
