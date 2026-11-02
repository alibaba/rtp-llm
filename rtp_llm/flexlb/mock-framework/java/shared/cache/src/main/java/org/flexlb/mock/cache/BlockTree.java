package org.flexlb.mock.cache;

import java.util.*;

/** RTP Full-KV prefix index. Every mutation runs under KvCache's monitor. */
final class BlockTree {
    static final class Node {
        final long key;
        long admission;
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
    private final NavigableSet<Node> deviceLeaves, hostLeaves;
    private long sequence, version;
    private java.util.function.Consumer<CacheMetadata.Change> observer = change -> {};
    void observe(java.util.function.Consumer<CacheMetadata.Change> observer) { this.observer = observer; }
    private static CacheMetadata.Medium medium(boolean host) {
        return host ? CacheMetadata.Medium.HOST : CacheMetadata.Medium.HBM;
    }
    private void changed(long key, CacheMetadata.Medium before, CacheMetadata.Medium after) {
        observer.accept(new CacheMetadata.Change(++version, key, before, after));
    }

    BlockTree(KvCache.EvictionPolicy devicePolicy, KvCache.EvictionPolicy hostPolicy) {
        deviceLeaves = new TreeSet<>(order(devicePolicy)); hostLeaves = new TreeSet<>(order(hostPolicy));
    }
    private static Comparator<Node> order(KvCache.EvictionPolicy policy) {
        Comparator<Node> order = switch (policy) {
            case LRU -> Comparator.comparingLong((Node n) -> n.access).thenComparingLong(n -> n.admission);
            case LFU -> Comparator.comparingLong((Node n) -> n.hits).thenComparingLong(n -> n.access);
            case FIFO -> Comparator.comparingLong((Node n) -> n.admission);
        };
        return order.thenComparingLong(n -> n.key);
    }
    List<Node> match(List<Long> keys) {
        var result = new ArrayList<Node>();
        var children = roots;
        long access = ++sequence;
        for (long key : keys) {
            var node = children.get(key);
            if (node == null) break;
            removeCandidate(node); node.access = access; node.hits++; refresh(node);
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
        changed(key, CacheMetadata.Medium.NONE, medium(host)); return node;
    }
    // Transactions pin the complete matched/published prefix, including every ancestor.
    void pin(Node node) { removeCandidate(node); node.pins++; }
    void unpin(Node node) {
        if (node.pins <= 0) throw new IllegalStateException("Unbalanced tree pin");
        node.pins--; refresh(node);
    }
    Node movableDevice() { return deviceLeaves.isEmpty() ? null : deviceLeaves.first(); }
    void relocate(Node node, BlockPool.Reference target, boolean host) {
        removeCandidate(node);
        if (node.parent != null) {
            removeCandidate(node.parent); node.parent.deviceChildren += host ? -1 : 1;
        }
        var previous = medium(node.host);
        node.cache.close(); node.cache = target; node.host = host; node.admission = ++sequence;
        refresh(node); if (node.parent != null) refresh(node.parent);
        changed(node.key, previous, medium(host));
    }
    boolean evict(boolean host) {
        var candidates = host ? hostLeaves : deviceLeaves;
        if (candidates.isEmpty()) return false;
        var node = candidates.first();
        if (node.parent != null) removeCandidate(node.parent);
        (node.parent == null ? roots : node.parent.children).remove(node.key);
        if (node.parent != null && !node.host) node.parent.deviceChildren--;
        // Full KV descendants cannot match once their prefix is removed, even across tiers.
        var pending = new ArrayDeque<Node>(); pending.add(node);
        while (!pending.isEmpty()) {
            var removed = pending.removeFirst(); pending.addAll(removed.children.values());
            removeCandidate(removed); nodes.remove(removed.key); removed.cache.close();
            changed(removed.key, medium(removed.host), CacheMetadata.Medium.NONE);
        }
        if (node.parent != null) refresh(node.parent);
        return true;
    }
    long version() { return version; }
    CacheMetadata.Snapshot metadata() {
        var blocks = new HashMap<Long, CacheMetadata.Medium>();
        nodes.forEach((key, node) -> blocks.put(key, medium(node.host)));
        return new CacheMetadata.Snapshot(version, blocks);
    }
    Set<Long> keys() { return Set.copyOf(nodes.keySet()); }
    void clear() {
        if (nodes.values().stream().anyMatch(n -> n.pins != 0)) throw new IllegalStateException("Cannot clear a pinned tree");
        for (var node : nodes.values()) {
            node.cache.close(); changed(node.key, medium(node.host), CacheMetadata.Medium.NONE);
        }
        nodes.clear(); roots.clear(); deviceLeaves.clear(); hostLeaves.clear();
    }
    private void removeCandidate(Node node) { deviceLeaves.remove(node); hostLeaves.remove(node); }
    private void refresh(Node node) {
        if (node.pins != 0) return;
        int sameTierChildren = node.host ? node.children.size() - node.deviceChildren : node.deviceChildren;
        if (sameTierChildren == 0) (node.host ? hostLeaves : deviceLeaves).add(node);
    }
}
