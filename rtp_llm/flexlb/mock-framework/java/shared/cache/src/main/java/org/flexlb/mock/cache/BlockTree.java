package org.flexlb.mock.cache;

import java.util.*;

/** RTP Full-KV prefix index. Every mutation runs under KvCache's monitor. */
final class BlockTree {
    static final class Node {
        final long key, admission;
        final Node parent;
        BlockPool.Reference cache;
        boolean host;
        final Map<Long, Node> children = new HashMap<>();
        long access, hits;
        int pins, deviceChildren;
        Node(long key, Node parent, BlockPool.Reference cache, boolean host, long sequence) {
            this.key = key; this.parent = parent; this.cache = cache; this.host = host;
            admission = access = sequence;
        }
    }
    private final Map<Long, Node> nodes = new HashMap<>(), roots = new HashMap<>();
    private final NavigableSet<Node> deviceLeaves, hostLeaves, deviceMovable;
    private long sequence, version;
    private int pinnedDevice;

    BlockTree(KvCache.EvictionPolicy devicePolicy, KvCache.EvictionPolicy hostPolicy) {
        deviceLeaves = new TreeSet<>(order(devicePolicy)); hostLeaves = new TreeSet<>(order(hostPolicy));
        deviceMovable = new TreeSet<>(order(devicePolicy));
    }
    private static Comparator<Node> order(KvCache.EvictionPolicy policy) {
        Comparator<Node> order = switch (policy) {
            case LRU -> Comparator.comparingLong((Node n) -> n.access).thenComparingLong(n -> n.admission);
            case LFU -> Comparator.comparingLong((Node n) -> n.hits).thenComparingLong(n -> n.access);
            case FIFO -> Comparator.comparingLong((Node n) -> n.admission).thenComparingLong(n -> n.access);
        };
        return order.thenComparingLong(n -> n.key);
    }
    List<Node> match(List<Long> keys) {
        var result = new ArrayList<Node>();
        var children = roots;
        for (long key : keys) {
            var node = children.get(key);
            if (node == null) break;
            removeCandidate(node); node.access = ++sequence; node.hits++; refresh(node);
            result.add(node); children = node.children;
        }
        return result;
    }
    void validatePath(List<Long> keys) {
        Long parent = null;
        for (long key : keys) {
            var existing = nodes.get(key);
            if (existing != null && !Objects.equals(existing.parent == null ? null : existing.parent.key, parent))
                throw new IllegalArgumentException("A prefix identity cannot belong to two paths");
            parent = key;
        }
    }
    Node find(long key) { return nodes.get(key); }
    Node insert(long key, Node parent, BlockPool.Reference cache, boolean host) {
        if (parent != null) { removeCandidate(parent); if (!host) parent.deviceChildren++; }
        var node = new Node(key, parent, cache, host, ++sequence);
        nodes.put(key, node); (parent == null ? roots : parent.children).put(key, node);
        refresh(node); if (parent != null) refresh(parent);
        version++; return node;
    }
    void pin(Node node) { removeCandidate(node); if (node.pins++ == 0 && !node.host) pinnedDevice++; }
    void unpin(Node node) {
        if (node.pins <= 0) throw new IllegalStateException("Unbalanced tree pin");
        if (--node.pins == 0 && !node.host) pinnedDevice--; refresh(node);
    }
    Node movableDevice() { return deviceMovable.isEmpty() ? null : deviceMovable.first(); }
    void relocate(Node node, BlockPool.Reference target, boolean host) {
        removeCandidate(node);
        if (node.parent != null) {
            removeCandidate(node.parent); node.parent.deviceChildren += host ? -1 : 1;
        }
        if (node.pins > 0) pinnedDevice += host ? -1 : 1;
        node.cache.close(); node.cache = target; node.host = host;
        refresh(node); if (node.parent != null) refresh(node.parent);
        // Moving tiers keeps the same advertised logical key set.
    }
    boolean evict(boolean host) {
        var candidates = host ? hostLeaves : deviceLeaves;
        if (candidates.isEmpty()) return false;
        var node = candidates.first(); removeCandidate(node);
        nodes.remove(node.key); (node.parent == null ? roots : node.parent.children).remove(node.key);
        node.cache.close(); version++;
        if (node.parent != null) {
            removeCandidate(node.parent); if (!host) node.parent.deviceChildren--; refresh(node.parent);
        }
        return true;
    }
    int pinnedDeviceBlocks() { return pinnedDevice; }
    long version() { return version; }
    Set<Long> keys() { return Set.copyOf(nodes.keySet()); }
    void clear() {
        if (nodes.values().stream().anyMatch(n -> n.pins != 0)) throw new IllegalStateException("Cannot clear a pinned tree");
        for (var node : nodes.values()) node.cache.close();
        version += nodes.size(); nodes.clear(); roots.clear(); deviceLeaves.clear(); hostLeaves.clear(); deviceMovable.clear();
    }
    private void removeCandidate(Node node) { deviceLeaves.remove(node); hostLeaves.remove(node); deviceMovable.remove(node); }
    private void refresh(Node node) {
        if (node.pins != 0) return;
        if (node.children.isEmpty()) (node.host ? hostLeaves : deviceLeaves).add(node);
        if (!node.host && node.deviceChildren == 0) deviceMovable.add(node);
    }
}
