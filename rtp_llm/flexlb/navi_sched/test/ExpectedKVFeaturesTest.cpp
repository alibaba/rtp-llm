#include "aios/navi_sched/data/NaviSchedLatencyPrediction.h"
#include "aios/navi_sched/resource/NonLinearPrefillModelR.h"
#include "autil/DataBuffer.h"
#include "unittest/unittest.h"
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <iostream>
#include <limits>
namespace navi_sched {
namespace {
NaviSchedLatencyFeatures baseFeatures() {
    NaviSchedLatencyFeatures f;
    EXPECT_TRUE(f.appendCapacitySplitAggregate(2,2,192512,115764,1.0e10,2.0e10,0,0,0,0,0));
    EXPECT_TRUE(f.appendPrecedingRequestCount(2));return f;
}
NaviSchedExpectedKVWorkload expected() {
    NaviSchedExpectedKVWorkload e;e.running={0.5,6144,420731.5,0.5*841463.0*841463.0,0.5*12288.0*841463.0};e.precedingRequests=0.5;return e;
}
std::vector<double> model() {
    std::vector<double> p(19,0.01);p[7]=1.0;p[8]=1.0;p[15]=5.0/NonLinearPrefillModelR::PARAMETER_SCALES[15];return p;
}
TEST(ExpectedKVFeaturesTest, FactualIntegersUnchangedAndExpectedFractionPreserved) {
    auto f=baseFeatures();ASSERT_TRUE(f.setExpectedKVWorkload(expected()));
    EXPECT_EQ(2,f.getPrecedingRequestCount());EXPECT_DOUBLE_EQ(2.5,f.getPlanningPrecedingRequestCount());
    EXPECT_EQ(192512,f.getCacheHitTokenCount());EXPECT_DOUBLE_EQ(198656.0,f.getPlanningCacheHitTokenCount());
    EXPECT_EQ(115764,f.getCacheMissTokenCount());EXPECT_DOUBLE_EQ(536495.5,f.getPlanningCacheMissTokenCount());
    EXPECT_EQ(2,f.getBatchCount());
}
TEST(ExpectedKVFeaturesTest, OnlyPredictionEncodingCarriesExpectedWorkload) {
    auto raw=baseFeatures();auto planned=raw;ASSERT_TRUE(planned.setExpectedKVWorkload(expected()));
    autil::DataBuffer a,b;raw.serialize(a);planned.serialize(b);
    EXPECT_EQ(std::string(a.getData(),a.getDataLen()),std::string(b.getData(),b.getDataLen()));
    NaviSchedLatencyPrediction p;p.features=planned;p.floorFeatures=raw;p.recomputable=true;p.hasFloor=true;p.fixedLatencyMs=-3.5;
    autil::DataBuffer wire;p.serialize(wire);wire.write(int64_t(77881));
    NaviSchedLatencyPrediction d;d.deserialize(wire);int64_t sentinel=0;wire.read(sentinel);
    ASSERT_TRUE(d.valid());ASSERT_NE(nullptr,d.features.getExpectedKVWorkload());EXPECT_EQ(nullptr,d.floorFeatures.getExpectedKVWorkload());
    EXPECT_EQ(77881,sentinel);EXPECT_DOUBLE_EQ(-3.5,d.fixedLatencyMs);EXPECT_TRUE(d.hasFloor);
    EXPECT_DOUBLE_EQ(0.5,d.features.getExpectedKVWorkload()->running[0]);
    const auto parameters=model();EXPECT_DOUBLE_EQ(NonLinearPrefillModelR::predict(p.features,parameters),NonLinearPrefillModelR::predict(d.features,parameters));
}
TEST(ExpectedKVFeaturesTest, BothPredictionAndFloorHaveIndependentExpectedComponents) {
    auto a=baseFeatures(),b=baseFeatures();auto e=expected();ASSERT_TRUE(a.setExpectedKVWorkload(e));e.running[2]=123.25;ASSERT_TRUE(b.setExpectedKVWorkload(e));
    NaviSchedLatencyPrediction p;p.features=a;p.floorFeatures=b;p.recomputable=true;autil::DataBuffer wire;p.serialize(wire);NaviSchedLatencyPrediction d;d.deserialize(wire);
    ASSERT_NE(nullptr,d.features.getExpectedKVWorkload());ASSERT_NE(nullptr,d.floorFeatures.getExpectedKVWorkload());
    EXPECT_DOUBLE_EQ(420731.5,d.features.getExpectedKVWorkload()->running[2]);EXPECT_DOUBLE_EQ(123.25,d.floorFeatures.getExpectedKVWorkload()->running[2]);
}
TEST(ExpectedKVFeaturesTest, AppendAndSubtractCarryWholeExpectedComponentExactlyOnce) {
    auto baseline=baseFeatures(),planned=baseline;ASSERT_TRUE(planned.setExpectedKVWorkload(expected()));
    NaviSchedLatencyFeatures combined;ASSERT_TRUE(combined.append(planned));
    EXPECT_EQ(planned.getExpectedKVWorkload(),combined.getExpectedKVWorkload());
    ASSERT_TRUE(combined.append(baseline));EXPECT_DOUBLE_EQ(4.5,combined.getPlanningPrecedingRequestCount());
    ASSERT_TRUE(combined.subtract(planned));EXPECT_EQ(nullptr,combined.getExpectedKVWorkload());EXPECT_EQ(2,combined.getPrecedingRequestCount());
    auto duplicate=planned;ASSERT_TRUE(duplicate.append(planned));EXPECT_DOUBLE_EQ(1.0,duplicate.getExpectedKVWorkload()->precedingRequests);
    EXPECT_DOUBLE_EQ(0.5,planned.getExpectedKVWorkload()->precedingRequests);
    ASSERT_TRUE(duplicate.subtract(planned));EXPECT_DOUBLE_EQ(0.5,duplicate.getExpectedKVWorkload()->precedingRequests);
}
TEST(ExpectedKVFeaturesTest, InvalidExpectedComponentRejectedAndSampleDecodeClearsIt) {
    NaviSchedLatencyFeatures legacy;auto e=expected();EXPECT_FALSE(legacy.setExpectedKVWorkload(e));
    auto f=baseFeatures();e.running[1]=-1;EXPECT_FALSE(f.setExpectedKVWorkload(e));e=expected();e.waiting[4]=std::numeric_limits<double>::quiet_NaN();EXPECT_FALSE(f.setExpectedKVWorkload(e));
    ASSERT_TRUE(f.setExpectedKVWorkload(expected()));autil::DataBuffer wire;baseFeatures().serialize(wire);f.deserialize(wire);EXPECT_EQ(nullptr,f.getExpectedKVWorkload());
    ASSERT_TRUE(f.setExpectedKVWorkload({}));EXPECT_EQ(nullptr,f.getExpectedKVWorkload());
}
TEST(ExpectedKVFeaturesTest, ExistingModelCostHelpersSeeExactlyTheSameExpectedInput) {
    auto f=baseFeatures();ASSERT_TRUE(f.setExpectedKVWorkload(expected()));const auto p=model();ASSERT_TRUE(NonLinearPrefillModelR::validateParameters(p));
    const auto cost=NonLinearPrefillModelR::calculateLinearCost(p,f)-p[6]*NonLinearPrefillModelR::PARAMETER_SCALES[6];
    const auto direct=NonLinearPrefillModelR::calculateLatencyFromRequestCostSum(p,cost,f.getPlanningCacheHitTokenCount()+f.getPlanningCacheMissTokenCount(),f.getPlanningPrecedingRequestCount());
    EXPECT_NEAR(NonLinearPrefillModelR::predict(f,p),direct,1e-8);
}
class ExpectedKVCorpusTest : public TESTBASE {};
TEST_F(ExpectedKVCorpusTest, CorpusSamplesSurviveAppendAndWireRoundTrip) {
    const auto overridePath=std::getenv("NAVI_KV_RAMP_CORPUS");
    const std::string path=overridePath ? overridePath : GET_PRIVATE_TEST_DATA_PATH()+"/kv_preparation/synthetic_ramp_reference.txt";
    std::ifstream input(path);ASSERT_TRUE(input.good());std::string line;size_t count=0;double maximum=0;
    while(std::getline(input,line)) {
        std::istringstream row(line);std::string bid;row>>bid;double a[15];for(auto &v:a)row>>v;std::vector<double> p(19);for(auto &v:p)row>>v;
        double baseline,residual,queue,answer;size_t pending;row>>baseline>>residual>>queue>>answer>>pending;
        NaviSchedLatencyFeatures raw;ASSERT_TRUE(raw.appendCapacitySplitAggregate(a[0],a[1],a[2],a[3],a[4],a[5],a[6],a[7],a[8],a[9],a[10]));ASSERT_TRUE(raw.appendPrecedingRequestCount(a[11]));ASSERT_TRUE(raw.appendRunningElapsedTimeMs(a[12]));ASSERT_TRUE(raw.appendRunningDecodeProgress(a[13],a[14]));
        ASSERT_NEAR(baseline,NonLinearPrefillModelR::predict(raw,p),1e-5)<<bid;
        NaviSchedExpectedKVWorkload e;
        for(size_t i=0;i<pending;++i){double h,n,weight;row>>h>>n>>weight;e.precedingRequests+=weight;e.running[0]+=weight;e.running[1]+=weight*h;e.running[2]+=weight*n;e.running[3]+=weight*n*n;e.running[4]+=weight*h*n;}
        ASSERT_FALSE(row.fail());ASSERT_TRUE(raw.setExpectedKVWorkload(e));
        NaviSchedLatencyFeatures appended;ASSERT_TRUE(appended.append(raw));
        NaviSchedLatencyPrediction prediction;prediction.features=appended;prediction.recomputable=true;
        autil::DataBuffer wire;prediction.serialize(wire);NaviSchedLatencyPrediction decoded;decoded.deserialize(wire);ASSERT_TRUE(decoded.valid());
        const auto value=NonLinearPrefillModelR::predict(decoded.features,p)+residual+queue;
        maximum=std::max(maximum,std::abs(value-answer));ASSERT_NEAR(answer,value,1e-5)<<bid;
        EXPECT_EQ(int64_t(a[0]),decoded.features.getBatchCount());++count;
    }
    EXPECT_EQ(overridePath ? 1331u : 64u,count);std::cout<<"expected_features_wire_replay_max_ms="<<maximum<<std::endl;
}
}
}
