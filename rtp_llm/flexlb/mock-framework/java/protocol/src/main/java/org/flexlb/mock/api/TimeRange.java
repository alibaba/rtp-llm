package org.flexlb.mock.api;

public sealed interface TimeRange permits TimeRange.All, TimeRange.Latest, TimeRange.Interval {
    record All() implements TimeRange {}
    record Latest() implements TimeRange {}
    record Interval(Duration start, Duration end) implements TimeRange {}
    record Duration(String value) {}
}
