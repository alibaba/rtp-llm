package org.flexlb.balance.endpoint.prefill;

import org.flexlb.balance.prediction.PrefillBatchFeatures;

/**
 * 一个 BATCH 的最后一个成员结束后的汇总，用来打日志和在线校准耗时预测。
 *
 * @param batchId            batch 编号
 * @param featuresAtCommit   提交时按成员算出的 batch 特征
 * @param predictedWorkMs    提交时的预测耗时
 * @param actualWorkMs       引擎上报的成员执行耗时中的最大值
 * @param anyMemberSucceeded 是否至少有一个成员由引擎上报成功
 * @param usableForLearning  是否所有成员都由引擎上报成功；只有这时才拿去校准预测
 */
public record FinishedBatch(long batchId,
                            PrefillBatchFeatures featuresAtCommit,
                            long predictedWorkMs,
                            long actualWorkMs,
                            boolean anyMemberSucceeded,
                            boolean usableForLearning) {
}
