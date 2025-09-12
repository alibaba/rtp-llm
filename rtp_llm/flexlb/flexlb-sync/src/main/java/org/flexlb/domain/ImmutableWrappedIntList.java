package org.flexlb.domain;

import java.util.List;

/**
 * @author zjw
 * description:
 * date: 2025/4/23
 */
public class ImmutableWrappedIntList implements ImmutableIntList {

    private final List<Integer> intList;

    public ImmutableWrappedIntList(List<Integer> intList) {
        this.intList = intList;
    }

    @Override
    public int size() {
        return intList.size();
    }

    @Override
    public int get(int idx) {
        return intList.get(idx);
    }

    @Override
    public int[] toArray() {
        return intList.stream().mapToInt(Integer::intValue).toArray();
    }
}
