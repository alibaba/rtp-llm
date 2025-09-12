package org.flexlb.metric;

import java.util.List;

import com.taobao.kmonitor.ImmutableMetricTags;
import com.taobao.kmonitor.KMonitor;
import com.taobao.kmonitor.MetricType;
import com.taobao.kmonitor.PriorityType;
import com.taobao.kmonitor.core.MetricsData;
import com.taobao.kmonitor.core.MetricsRecords;
import com.taobao.kmonitor.core.MetricsTags;

public class EmptyKmonitor implements KMonitor {
    @Override
    public boolean register(String metricName, MetricType metricType) {
        return false;
    }

    @Override
    public boolean register(String metricName, MetricType metricType, PriorityType priorityType) {
        return false;
    }

    @Override
    public boolean register(String metricName, MetricType metricType, int statisticsType) {
        return false;
    }

    @Override
    public boolean register(String metricName, MetricType metricType, int statisticsType, PriorityType priorityType) {
        return false;
    }

    @Override
    public MetricsData registerMetric(String metricName, MetricType metricType, int statisticsType, PriorityType priorityType) {
        return null;
    }

    @Override
    public MetricsData registerMetric(String metricName, MetricType metricType) {
        return null;
    }

    @Override
    public void report(String metricName, double value) {

    }

    @Override
    public void report(String metricName, MetricsTags metricsTags, double value) {

    }

    @Override
    public void report(String metricName, MetricsTags metricsTags, double value, Long timestamp) {

    }

    @Override
    public void recycle(String metricName, MetricsTags metricsTags) {

    }

    @Override
    public void unregister(String metricName) {

    }

    @Override
    public void putTag(String key, String value) {

    }

    @Override
    public String name() {
        return null;
    }

    @Override
    public MetricsRecords getMetrics(PriorityType priorityType, boolean all) {
        return null;
    }

    @Override
    public ImmutableMetricTags getTags() {
        return null;
    }

    @Override
    public void close() {

    }

    @Override
    public int getPriority() {
        return 0;
    }

    @Override
    public List<Class<?>> getSinkTypes() {
        return null;
    }

    @Override
    public void setSinkTypes(List<Class<?>> list) {

    }
}
