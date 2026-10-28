package org.flexlb.mock.engine.rtp.cache;

import java.util.*;
import org.flexlb.mock.cache.BlockPool;

/** RTP full-attention device tree. Called only under DeviceCache's owner monitor. */
final class BlockTree {
    static final class Node {
        final long key, admission;
        final Node parent;
        final BlockPool.Reference cache;
        final Map<Long, Node> children = new HashMap<>();
        long access, hits;
        int pins;
        Node(long key, Node parent, BlockPool.Reference cache, long sequence) {
            this.key = key; this.parent = parent; this.cache = cache; admission = access = sequence;
        }
    }
    private final Map<Long, Node> nodes = new HashMap<>(), roots = new HashMap<>();
    private final NavigableSet<Node> candidates;
    private long sequence, version;
    private int pinned;

    BlockTree(DeviceCache.EvictionPolicy policy) {
        Comparator<Node> order = switch (policy) {
            case LRU -> Comparator.comparingLong((Node n) -> n.access).thenComparingLong(n -> n.admission);
            case LFU -> Comparator.comparingLong((Node n) -> n.hits).thenComparingLong(n -> n.access);
            case FIFO -> Comparator.comparingLong((Node n) -> n.admission).thenComparingLong(n -> n.access);
        };
        candidates = new TreeSet<>(order.thenComparingLong(n -> n.key));
    }
    List<Node> match(List<Long> keys) {
        var result = new ArrayList<Node>();
        var children = roots;
        for (long key : keys) {
            var node = children.get(key);
            if (node == null) break;
            candidates.remove(node);
            node.access = ++sequence; node.hits++;
            refresh(node);
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
    Node insert(long key, Node parent, BlockPool.Reference cache) {
        if (parent != null) candidates.remove(parent);
        var node = new Node(key, parent, cache, ++sequence);
        nodes.put(key, node);
        (parent == null ? roots : parent.children).put(key, node);
        version++;
        return node;
    }
    void pin(Node node) {
        candidates.remove(node);
        if (node.pins++ == 0) pinned++;
    }
    void unpin(Node node) {
        if (node.pins <= 0) throw new IllegalStateException("Unbalanced tree pin");
        if (--node.pins == 0) pinned--;
        refresh(node); // Release is not an access and must not update LRU/LFU heat.
    }
    boolean evict() {
        var node = candidates.pollFirst();
        if (node == null) return false;
        nodes.remove(node.key);
        (node.parent == null ? roots : node.parent.children).remove(node.key);
        node.cache.close(); version++;
        if (node.parent != null) refresh(node.parent);
        return true;
    }
    int pinnedBlocks() { return pinned; }
    long version() { return version; }
    Set<Long> keys() { return Set.copyOf(nodes.keySet()); }
    void clear() {
        if (pinned != 0) throw new IllegalStateException("Cannot clear a pinned tree");
        for (var node : nodes.values()) node.cache.close();
        version += nodes.size(); nodes.clear(); roots.clear(); candidates.clear();
    }
    private void refresh(Node node) {
        if (node.pins == 0 && node.children.isEmpty()) candidates.add(node);
    }
}
