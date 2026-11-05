package org.flexlb.mock.engine.cache;

import java.util.*;

/** RTP Full-KV prefix index. Every mutation runs under KvCache's monitor. */
final class BlockTree {
    static final class Node {
        final long key, admission, insertedNanos = System.nanoTime();
        final Node parent;
        final BlockPool.Reference cache;
        final Map<Long, Node> children = new HashMap<>();
        long access;

        Node(long key, Node parent, BlockPool.Reference cache, long sequence) {
            this.key = key;
            this.parent = parent;
            this.cache = cache;
            admission = access = sequence;
        }
    }

    private final Map<Long, Node> nodes = new HashMap<>(), roots = new HashMap<>();
    private final NavigableSet<Node> leaves = new TreeSet<>(Comparator
        .comparingLong((Node node) -> node.access).thenComparingLong(node -> node.admission));
    private long sequence, version;
    private java.util.function.Consumer<CacheMetadata.Change> observer = change -> {};
    void observe(java.util.function.Consumer<CacheMetadata.Change> observer) { this.observer = observer; }
    private void changed(long key, boolean added) {
        observer.accept(new CacheMetadata.Change(++version, key, added));
    }

    List<Node> match(List<Long> keys) {
        var result = new ArrayList<Node>();
        var children = roots;
        long access = ++sequence;
        for (long key : keys) {
            var node = children.get(key);
            if (node == null) break;
            if (leaves.remove(node)) { node.access = access; leaves.add(node); }
            else node.access = access;
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
        if (parent != null) leaves.remove(parent);
        var node = new Node(key, parent, cache, ++sequence);
        nodes.put(key, node);
        (parent == null ? roots : parent.children).put(key, node);
        leaves.add(node);
        changed(key, true);
        return node;
    }

    Node evict() {
        for (var iterator = leaves.iterator(); iterator.hasNext();) {
            var node = iterator.next();
            if (node.cache.inUse()) continue;
            iterator.remove();
            nodes.remove(node.key);
            (node.parent == null ? roots : node.parent.children).remove(node.key);
            node.cache.close();
            if (node.parent != null && node.parent.children.isEmpty()) leaves.add(node.parent);
            changed(node.key, false);
            return node;
        }
        return null;
    }

    int size() { return nodes.size(); }
    long version() { return version; }
    CacheMetadata.SnapshotKeys metadataKeys() {
        return new CacheMetadata.SnapshotKeys(version, nodes.keySet().stream().mapToLong(Long::longValue).toArray());
    }
    Set<Long> keys() { return Set.copyOf(nodes.keySet()); }
    void clear() {
        for (var node : nodes.values()) {
            node.cache.close(); changed(node.key, false);
        }
        nodes.clear(); roots.clear(); leaves.clear();
    }
}
