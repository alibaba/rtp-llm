package org.flexlb.balance.strategy;

import java.util.Map;
import java.util.Queue;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedDeque;
import java.util.concurrent.locks.ReadWriteLock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

public class TaskLRU<K, V> {
    private final int capacity;
    private final Map<K, V> internalCache; // 注意命名
    private final Queue<K> trackingQueue;  // 按功能命名
    private final ReadWriteLock lock;

    public TaskLRU(int capacity) {
        this.capacity = capacity;
        this.internalCache = new ConcurrentHashMap<>(capacity);
        this.trackingQueue = new ConcurrentLinkedDeque<>();
        this.lock = new ReentrantReadWriteLock();
    }

    public V get(K key) {
        lock.readLock().lock();
        try {
            V val = internalCache.get(key);
            if (val != null) {
                if (trackingQueue.remove(key)) {
                    trackingQueue.offer(key);
                }
            }

            return val;
        } finally {
            lock.readLock().unlock();
        }
    }

    // return null if key not exists, otherwise return old value
    public V put(K key, V val) {
        lock.writeLock().lock();
        try {
            V oldVal = internalCache.get(key);
            if (oldVal != null) {
                internalCache.put(key, val); // update cache entry
                trackingQueue.remove(key);
                trackingQueue.offer(key);
                return oldVal;
            } else {
                if (internalCache.size() == capacity) {
                    K deletedKey = trackingQueue.poll();
                    internalCache.remove(deletedKey);
                }
                internalCache.put(key, val);
                trackingQueue.offer(key);
                return null;
            }
        } finally {
            lock.writeLock().unlock();
        }
    }

    public V remove(K key) {
        lock.writeLock().lock();
        try {
            V val = internalCache.remove(key);
            trackingQueue.remove(key);
            return val;
        } finally {
            lock.writeLock().unlock();
        }
    }

}