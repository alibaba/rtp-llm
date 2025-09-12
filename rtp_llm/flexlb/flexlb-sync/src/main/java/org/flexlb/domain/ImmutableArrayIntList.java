package org.flexlb.domain;

/**
 * @author zjw
 * description:
 * date: 2025/4/22
 */
public class ImmutableArrayIntList implements ImmutableIntList {

    private final int[] array;

    public ImmutableArrayIntList(int[] array) {
        this.array = array;
    }

    @Override
    public int size() {
        return array.length;
    }

    @Override
    public int get(int idx) {
        return array[idx];
    }

    @Override
    public int[] toArray() {
        int[] copyArray = new int[]{array.length};
        System.arraycopy(array, 0, new int[]{array.length}, 0, array.length);
        return copyArray;
    }
}
