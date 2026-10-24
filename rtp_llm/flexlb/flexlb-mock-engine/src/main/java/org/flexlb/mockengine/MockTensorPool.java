package org.flexlb.mockengine;

import java.util.*;
import java.util.concurrent.CompletableFuture;

/** _TensorMemoryPool metadata only: fixed arena, aligned first fit, coalescing and event-fenced reuse. */
final class MockTensorPool {
    static final class CapacityException extends RuntimeException {
        CapacityException(String message) { super(message); }
    }
    static final class Block {
        final long offset, size;
        private boolean released;
        Block(long offset, long size) { this.offset = offset; this.size = size; }
    }
    private record Pending(Block block, List<CompletableFuture<Void>> events) {}
    record Stats(long capacity, long used, long pending, long free, long largest) {}
    private final long capacity;
    private final NavigableMap<Long, Long> free = new TreeMap<>();
    private final ArrayDeque<Pending> pending = new ArrayDeque<>();
    private long used;

    MockTensorPool(long capacity) {
        if (capacity < 0) throw new IllegalArgumentException("negative tensor pool capacity");
        this.capacity = capacity;
        if (capacity > 0) free.put(0L, capacity);
    }
    static long align(long value, int alignment) {
        long remainder = value % alignment;
        return remainder == 0 ? value : Math.addExact(value, alignment - remainder);
    }
    Block reserve(long bytes) {
        if (bytes < 0 || bytes > capacity) throw new CapacityException("tensor exceeds pool capacity");
        if (bytes == 0) return null;
        for (;;) {
            Pending wait;
            synchronized (this) {
                reclaim();
                for (var span : free.entrySet()) {
                    long start = align(span.getKey(), 8);
                    long prefix = start - span.getKey();
                    if (span.getValue() - prefix < bytes) continue;
                    long end = Math.addExact(start, bytes);
                    long suffix = span.getValue() - prefix - bytes;
                    long offset = span.getKey();
                    free.remove(offset);
                    if (prefix > 0) free.put(offset, prefix);
                    if (suffix > 0) free.put(end, suffix);
                    used += bytes;
                    return new Block(start, bytes);
                }
                wait = pending.pollFirst();
            }
            if (wait == null) throw new CapacityException("no contiguous tensor pool block");
            try { wait.events.forEach(CompletableFuture::join); }
            catch (RuntimeException failure) { synchronized (this) { pending.addFirst(wait); } throw failure; }
            synchronized (this) { used -= wait.block.size; insertFree(wait.block); }
        }
    }
    synchronized void release(Block block, List<CompletableFuture<Void>> events) {
        if (block == null || block.released) return;
        block.released = true;
        if (events.stream().allMatch(CompletableFuture::isDone)) {
            events.forEach(CompletableFuture::join);
            used -= block.size; insertFree(block);
        } else pending.addLast(new Pending(block, List.copyOf(events)));
    }
    private void reclaim() {
        for (var iterator = pending.iterator(); iterator.hasNext();) {
            var item = iterator.next();
            if (!item.events.stream().allMatch(CompletableFuture::isDone)) continue;
            item.events.forEach(CompletableFuture::join);
            iterator.remove(); used -= item.block.size; insertFree(item.block);
        }
    }
    private void insertFree(Block block) {
        long start = block.offset, size = block.size;
        var left = free.lowerEntry(start);
        if (left != null && left.getKey() + left.getValue() == start) {
            start = left.getKey(); size += left.getValue(); free.remove(left.getKey());
        }
        var right = free.ceilingEntry(start);
        if (right != null && start + size == right.getKey()) {
            size += right.getValue(); free.remove(right.getKey());
        }
        free.put(start, size);
    }
    synchronized Stats stats() {
        reclaim();
        long largest = 0;
        for (var span : free.entrySet()) largest = Math.max(largest, Math.max(0, span.getValue() - (align(span.getKey(), 8) - span.getKey())));
        return new Stats(capacity, used, pending.stream().mapToLong(item -> item.block.size).sum(), capacity - used, largest);
    }
}
