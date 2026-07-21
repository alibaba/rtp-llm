package org.flexlb.metric;

public class FlexStatisticsType {

    public static final int SUMMARY = 1 << 7;

    public static boolean needSummary(int flag) {
        return (flag & SUMMARY) > 0;
    }
}
