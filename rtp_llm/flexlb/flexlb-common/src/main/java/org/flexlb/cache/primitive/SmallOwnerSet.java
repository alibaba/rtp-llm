package org.flexlb.cache.primitive;

import java.util.AbstractSet;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.Objects;

/** Readers see immutable owner arrays published by index mutations. */
public final class SmallOwnerSet extends AbstractSet<String> {
    private volatile String[] owners = new String[0];

    @Override
    public synchronized boolean add(String owner) {
        Objects.requireNonNull(owner);
        if (contains(owner)) {
            return false;
        }
        String[] updated = Arrays.copyOf(owners, owners.length + 1);
        updated[owners.length] = owner;
        owners = updated;
        return true;
    }

    @Override
    public synchronized boolean remove(Object owner) {
        for (int index = 0; index < owners.length; index++) {
            if (owners[index].equals(owner)) {
                String[] updated = new String[owners.length - 1];
                System.arraycopy(owners, 0, updated, 0, index);
                System.arraycopy(owners, index + 1, updated, index, updated.length - index);
                owners = updated;
                return true;
            }
        }
        return false;
    }

    @Override
    public boolean contains(Object owner) {
        for (String candidate : owners) {
            if (candidate.equals(owner)) {
                return true;
            }
        }
        return false;
    }

    @Override
    public int size() { return owners.length; }

    @Override
    public synchronized void clear() { owners = new String[0]; }

    @Override
    public Iterator<String> iterator() {
        Iterator<String> snapshot = List.of(owners).iterator();
        return new Iterator<>() {
            private String current;
            public boolean hasNext() { return snapshot.hasNext(); }
            public String next() { current = snapshot.next(); return current; }
            public void remove() {
                if (current == null) {
                    throw new IllegalStateException();
                }
                SmallOwnerSet.this.remove(current);
                current = null;
            }
        };
    }
}
