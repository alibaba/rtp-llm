package org.flexlb.util;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * 过滤只读快照；全部元素通过时复用原列表，首次排除元素时才创建结果列表。
 * <p>
 * 原列表在筛选和返回后都不能再修改。调用方按索引顺序对每个元素调用一次 {@link #keepIf}，
 * 全部处理完再调用 {@link #toList()}，之后不再使用此筛选器。
 */
public final class ReadOnlyListFilter<T> {

    private final List<T> source;
    // null 表示此前所有元素都通过，仍可直接返回原快照，并非没有匹配元素。
    private List<T> filtered;

    public ReadOnlyListFilter(List<T> source) {
        this.source = source;
    }

    /**
     * 决定是否保留当前索引的元素。索引必须从 0 开始依次递增，不跳过或重复处理。
     */
    public void keepIf(int index, boolean keep) {
        if (keep) {
            if (filtered != null) {
                filtered.add(source.get(index));
            }
        } else if (filtered == null) {
            filtered = new ArrayList<>(source.size() - 1);
            // 第一次排除元素：它前面的元素都已通过，复制这段前缀即可，当前元素不加入结果。
            for (int previous = 0; previous < index; previous++) {
                filtered.add(source.get(previous));
            }
        }
    }

    /**
     * 筛选完成后返回只读结果。新列表仅由本筛选器持有，停止写入后包装即可，无需再复制数组。
     */
    public List<T> toList() {
        return filtered == null ? source : Collections.unmodifiableList(filtered);
    }
}
