package org.flexlb.metric;

public class FlexStatisticsType {

    public static final int MIN = 1 << 1;
    public static final int MAX = 1 << 2;
    public static final int PERCENTILE_75 = 1 << 3;
    public static final int PERCENTILE_95 = 1 << 4;
    public static final int PERCENTILE_99 = 1 << 5;
    public static final int SUM = 1 << 6;
    public static final int SUMMARY = 1 << 7;

    public static boolean needMaxMin(int flag) {
        return (flag & MIN) > 0 || (flag & MAX) > 0;
    }

    public static boolean needPercentile(int flag) {
        return (flag & PERCENTILE_75) > 0 || (flag & PERCENTILE_95) > 0 || (flag & PERCENTILE_99) > 0;
    }

    public static boolean needSum(int flag) {
        return (flag & SUM) > 0;
    }

    public static boolean needSummary(int flag) {
        return (flag & SUMMARY) > 0;
    }
}
