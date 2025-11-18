package org.flexlb.metric;

import lombok.Getter;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

/**
 * FlexMetricTags - 监控指标标签接口
 *
 * @author saichen.sm
 */
public interface FlexMetricTags {

    /**
     * 获取标签的键值对映射
     *
     * @return 标签键值对Map
     */
    Map<String, String> getTags();

    /**
     * 检查标签是否为空
     *
     * @return true表示为空，false表示不为空
     */
    boolean isEmpty();

    /**
     * 不可变标签实现类
     * 用于替代KMonitor的ImmutableMetricTags
     */
    class ImmutableFlexMetricTags implements FlexMetricTags {
        @Getter
        private final Map<String, String> tags;

        /**
         * 构造函数 - 通过Map创建
         *
         * @param tags 标签Map
         */
        public ImmutableFlexMetricTags(Map<String, String> tags) {
            if (tags == null || tags.isEmpty()) {
                this.tags = Collections.emptyMap();
            } else {
                this.tags = Collections.unmodifiableMap(new java.util.HashMap<>(tags));
            }
        }

        /**
         * 构造函数 - 通过键值对创建（支持多个键值对）
         *
         * @param keyValues 键值对，格式为key1, value1, key2, value2, ...
         */
        public ImmutableFlexMetricTags(String... keyValues) {
            if (keyValues == null || keyValues.length == 0) {
                this.tags = Collections.emptyMap();
            } else if (keyValues.length % 2 != 0) {
                throw new IllegalArgumentException("Key-value pairs must be even number of arguments");
            } else {
                Map<String, String> tempTags = new HashMap<>();
                for (int i = 0; i < keyValues.length; i += 2) {
                    String key = keyValues[i];
                    String value = keyValues[i + 1];
                    if (key != null && value != null) {
                        tempTags.put(key, value);
                    }
                }
                this.tags = Collections.unmodifiableMap(tempTags);
            }
        }

        @Override
        public boolean isEmpty() {
            return tags.isEmpty();
        }

        @Override
        public String toString() {
            return "ImmutableFlexMetricTags{" + "tags=" + tags + '}';
        }

        @Override
        public boolean equals(Object o) {
            if (this == o) {
                return true;
            }
            if (o == null || getClass() != o.getClass()) {
                return false;
            }
            ImmutableFlexMetricTags that = (ImmutableFlexMetricTags) o;
            return tags.equals(that.tags);
        }

        @Override
        public int hashCode() {
            return tags.hashCode();
        }
    }

    /**
     * 便捷工厂方法 - 创建不可变标签（通过Map）
     *
     * @param tags 标签Map
     * @return 不可变标签实例
     */
    static FlexMetricTags of(Map<String, String> tags) {
        return new ImmutableFlexMetricTags(tags);
    }

    /**
     * 便捷工厂方法 - 创建不可变标签（通过键值对）
     *
     * @param keyValues 键值对，格式为key1, value1, key2, value2, ...
     * @return 不可变标签实例
     */
    static FlexMetricTags of(String... keyValues) {
        return new ImmutableFlexMetricTags(keyValues);
    }
}
