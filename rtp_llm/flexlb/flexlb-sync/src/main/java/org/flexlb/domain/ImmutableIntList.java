package org.flexlb.domain;

/**
 * @author zjw
 * description:
 * date: 2025/4/22
 */
public interface ImmutableIntList {

    int size();

    int get(int idx);

    int[] toArray();

}
