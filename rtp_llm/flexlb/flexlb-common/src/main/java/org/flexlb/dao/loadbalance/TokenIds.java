package org.flexlb.dao.loadbalance;

import java.util.Objects;
import java.util.function.IntUnaryOperator;

/** Read-only, indexed access to prompt token IDs without requiring an array copy. */
public interface TokenIds {

    int size();

    int getInt(int index);

    static TokenIds wrap(int[] values) {
        return new ArrayTokenIds(values);
    }

    static TokenIds wrap(int size, IntUnaryOperator accessor) {
        if (size < 0) {
            throw new IllegalArgumentException("token ID size must not be negative");
        }
        return new IndexedTokenIds(size, accessor);
    }

    final class ArrayTokenIds implements TokenIds {
        private final int[] values;

        public ArrayTokenIds(int[] values) {
            this.values = Objects.requireNonNull(values, "token IDs must not be null");
        }

        @Override
        public int size() {
            return values.length;
        }

        @Override
        public int getInt(int index) {
            return values[index];
        }
    }

    final class IndexedTokenIds implements TokenIds {
        private final int size;
        private final IntUnaryOperator accessor;

        private IndexedTokenIds(int size, IntUnaryOperator accessor) {
            this.size = size;
            this.accessor = Objects.requireNonNull(accessor, "token ID accessor must not be null");
        }

        @Override
        public int size() {
            return size;
        }

        @Override
        public int getInt(int index) {
            if (index < 0 || index >= size) {
                throw new IndexOutOfBoundsException(index);
            }
            return accessor.applyAsInt(index);
        }
    }

}
