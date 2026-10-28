package org.flexlb.mock.cache;

import java.util.*;

/** Physical block accounting, independent of cache keys, engine, tier, or eviction policy. */
public final class BlockPool {
    public enum Kind { REQUEST, CACHE, LOAD, STORE, EVICTION }
    public record Snapshot(int total, int free, int available, int active) {}
    private final int capacity;
    private final int[] referenced = new int[Kind.values().length];
    private int allocated, active;
    private long generation;

    public BlockPool(int capacity) {
        if (capacity < 0) throw new IllegalArgumentException("Negative block capacity");
        this.capacity = capacity;
    }

    /** All or nothing. Eviction is the caller's responsibility. */
    public synchronized List<Reference> allocate(int count, Kind kind) {
        if (count < 0) throw new IllegalArgumentException("Negative block demand");
        Objects.requireNonNull(kind);
        if (count > capacity - allocated) return null;
        var result = new ArrayList<Reference>(count);
        for (int i = 0; i < count; i++) {
            var block = new Block();
            allocated++;
            result.add(add(block, kind));
        }
        return result;
    }

    /** Each owner receives a distinct reference; closing it cannot release another owner's pin. */
    public synchronized Reference retain(Reference source, Kind kind) {
        requireOwned(source);
        return add(source.block, Objects.requireNonNull(kind));
    }

    public synchronized Snapshot snapshot() {
        return new Snapshot(capacity, capacity - allocated, capacity - active, active);
    }
    public synchronized int referencedBlocks(Kind kind) { return referenced[kind.ordinal()]; }
    public synchronized long generation() { return generation; }
    public synchronized void awaitChange(long observed, long waitMs) throws InterruptedException {
        if (generation == observed) wait(Math.max(1, waitMs));
    }

    private Reference add(Block block, Kind kind) {
        if (block.refs[kind.ordinal()] == Integer.MAX_VALUE) throw new IllegalStateException("Block reference overflow");
        if (block.refs[kind.ordinal()]++ == 0) referenced[kind.ordinal()]++;
        if (kind != Kind.CACHE && block.activeRefs++ == 0) active++;
        block.totalRefs++;
        return new Reference(this, block, kind);
    }
    private synchronized void release(Reference reference) {
        if (reference.closed) return;
        reference.closed = true;
        var block = reference.block;
        if (--block.refs[reference.kind.ordinal()] == 0) referenced[reference.kind.ordinal()]--;
        boolean increased = false;
        if (reference.kind != Kind.CACHE && --block.activeRefs == 0) { active--; increased = true; }
        if (--block.totalRefs == 0) { allocated--; increased = true; }
        if (increased) { generation++; notifyAll(); }
    }
    private void requireOwned(Reference reference) {
        if (reference == null || reference.owner != this || reference.closed)
            throw new IllegalArgumentException("Reference is closed or belongs to another block pool");
    }
    private static final class Block {
        final int[] refs = new int[Kind.values().length];
        long totalRefs, activeRefs;
    }
    public static final class Reference implements AutoCloseable {
        private final BlockPool owner;
        private final Block block;
        private final Kind kind;
        private boolean closed;
        private Reference(BlockPool owner, Block block, Kind kind) {
            this.owner = owner; this.block = block; this.kind = kind;
        }
        @Override public void close() { owner.release(this); }
    }
}
