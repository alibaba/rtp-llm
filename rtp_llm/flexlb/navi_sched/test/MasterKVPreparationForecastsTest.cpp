#include "aios/navi_sched/resource/MasterKVPreparationForecasts.h"
#include "unittest/unittest.h"
#include <chrono>
#include <iostream>
#include <limits>

namespace navi_sched {
namespace {
using State=NativeKVExecutionState;
NaviSchedKVPreparation request(uint64_t id,uint64_t generation,State state,double age) {
    NaviSchedKVPreparation r{id,int64_t(id),generation,1000,3000,true,state};
    r.preparationElapsedMs=age;return r;
}
TEST(MasterKVPreparationModelTest, FindsNonnegativeFourFeatureSolutionAndRejectsInvalidSamples) {
    MasterKVPreparationModel m;EXPECT_FALSE(m.predict({1,1,0,0}));
    for(int repeat=0;repeat<2;++repeat) for(int i=0;i<32;++i) {
        MasterKVPreparationModel::Features x{1.0,double(i%7),double(i%5),double(i%3)};
        const double y=2.0+3*x[1]+5*x[2]+7*x[3];ASSERT_TRUE(m.record(x,y));
    }
    const auto p=m.coefficients();EXPECT_NEAR(2.0,p[0],1e-8);EXPECT_NEAR(3.0,p[1],1e-8);EXPECT_NEAR(5.0,p[2],1e-8);EXPECT_NEAR(7.0,p[3],1e-8);
    EXPECT_NEAR(2.0+3*9+5*8+7*6,*m.predict({1,9,8,6}),1e-8);
    const auto count=m.samples();EXPECT_FALSE(m.record({0,1,2,3},1));EXPECT_FALSE(m.record({1,-1,0,0},1));EXPECT_FALSE(m.record({1,1,1,1},-1));EXPECT_EQ(count,m.samples());
}
TEST(MasterKVPreparationModelTest, RankDeficiencyAndZeroDurationDoNotCreateNegativeOrFabricatedTime) {
    MasterKVPreparationModel m;ASSERT_TRUE(m.record({1,0,0,0},0));EXPECT_FALSE(m.predict({1,0,0,0}));
    ASSERT_TRUE(m.record({1,0,0,0},20));EXPECT_DOUBLE_EQ(10.0,*m.predict({1,0,0,0}));
    for(auto x:m.coefficients()){EXPECT_TRUE(std::isfinite(x));EXPECT_GE(x,0.0);}
}
TEST(MasterKVPreparationForecastsTest, PredictionIsFrozenPerLoadAndAgeSaturatesAtOne) {
    MasterKVPreparationForecasts f;f.beginSession(10);ASSERT_TRUE(f.reserve(1,4096,0));
    ASSERT_TRUE(f.observe(10,1,{request(1,1,State::PREPARING,0)},1000));
    auto first=f.snapshot({request(1,1,State::PREPARING,0)});ASSERT_EQ(1u,first.size());EXPECT_FALSE(first[0].totalMs);EXPECT_DOUBLE_EQ(1.0,first[0].weight(1000,0));
    ASSERT_TRUE(f.observe(10,2,{request(1,1,State::READY,100)},101000));EXPECT_EQ(1u,f.model().samples());
    ASSERT_TRUE(f.reserve(2,4096,0));ASSERT_TRUE(f.observe(10,3,{request(2,2,State::PREPARING,25)},200000));
    auto before=f.snapshot({request(2,2,State::PREPARING,25)});ASSERT_EQ(1u,before.size());ASSERT_TRUE(before[0].totalMs);const auto total=*before[0].totalMs;
    EXPECT_NEAR(100.0,total,1e-8);EXPECT_NEAR(0.25,before[0].weight(200000,0),1e-8);
    EXPECT_NEAR(0.5,before[0].weight(225000,0),1e-8);EXPECT_DOUBLE_EQ(1.0,before[0].weight(275000,0));EXPECT_DOUBLE_EQ(1.0,before[0].weight(400000,0));
    ASSERT_TRUE(f.reserve(3,4096,0));ASSERT_TRUE(f.observe(10,4,{request(2,2,State::PREPARING,30),request(3,3,State::PREPARING,0)},205000));
    ASSERT_TRUE(f.observe(10,5,{request(2,2,State::PREPARING,35),request(3,3,State::READY,500)},210000));
    const auto after=f.snapshot({request(2,2,State::PREPARING,35)});ASSERT_EQ(1u,after.size());EXPECT_DOUBLE_EQ(total,*after[0].totalMs);EXPECT_EQ(2u,f.model().samples());
}
TEST(MasterKVPreparationForecastsTest, DuplicateAndOutOfOrderReportsCannotRetrainOrResetAge) {
    MasterKVPreparationForecasts f;f.beginSession(10);ASSERT_TRUE(f.reserve(1,1000,1));
    const auto preparing=request(1,1,State::PREPARING,10);ASSERT_TRUE(f.observe(10,1,{preparing},1000));
    ASSERT_TRUE(f.observe(10,2,{request(1,1,State::READY,30)},21000));EXPECT_EQ(1u,f.model().samples());
    EXPECT_FALSE(f.observe(10,1,{preparing},22000));EXPECT_FALSE(f.observe(10,2,{preparing},22000));
    ASSERT_TRUE(f.observe(10,3,{request(1,1,State::READY,30)},23000));EXPECT_EQ(1u,f.model().samples());
}
TEST(MasterKVPreparationForecastsTest, FailureCancellationAndSessionChangeDoNotTrainSuccess) {
    MasterKVPreparationForecasts f;f.beginSession(10);ASSERT_TRUE(f.reserve(1,1000,1));
    ASSERT_TRUE(f.observe(10,1,{request(1,1,State::PREPARING,0)},1000));ASSERT_TRUE(f.observe(10,2,{request(1,1,State::FAILED,-1)},2000));EXPECT_EQ(0u,f.model().samples());
    ASSERT_TRUE(f.observe(10,3,{request(1,1,State::EXECUTING,-1)},3000));EXPECT_EQ(0u,f.model().samples());
    f.forget(1);EXPECT_EQ(0u,f.activeLoads());EXPECT_EQ(0u,f.reservations());
    ASSERT_TRUE(f.reserve(2,2000,2));f.beginSession(11);EXPECT_EQ(0u,f.reservations());EXPECT_EQ(0u,f.model().samples());
    EXPECT_FALSE(f.observe(10,4,{request(2,2,State::READY,10)},4000));
}
TEST(MasterKVPreparationForecastsTest, NewGenerationAndMissingBeginNeverUseOldForecastOutcome) {
    MasterKVPreparationForecasts f;f.beginSession(10);ASSERT_TRUE(f.reserve(1,1000,0));
    ASSERT_TRUE(f.observe(10,1,{request(1,1,State::PREPARING,0)},1000));
    ASSERT_TRUE(f.observe(10,2,{request(1,2,State::PREPARING,0)},2000));
    ASSERT_TRUE(f.observe(10,3,{request(1,1,State::READY,900)},3000));EXPECT_EQ(0u,f.model().samples());
    auto current=f.snapshot({request(1,2,State::PREPARING,0)});ASSERT_EQ(1u,current.size());EXPECT_EQ(2u,current[0].generation);
    ASSERT_TRUE(f.reserve(3,3000,0));ASSERT_TRUE(f.observe(10,4,{request(3,3,State::READY,20)},4000));EXPECT_EQ(0u,f.model().samples());
}
TEST(MasterKVPreparationForecastsTest, SameReportStartDoesNotTrainFromItsCompletionAndSameSessionKeepsState) {
    MasterKVPreparationForecasts f;f.beginSession(10);ASSERT_TRUE(f.reserve(1,1000,0));ASSERT_TRUE(f.reserve(2,1000,0));
    ASSERT_TRUE(f.observe(10,1,{request(1,1,State::PREPARING,0)},1000));
    ASSERT_TRUE(f.observe(10,2,{request(1,1,State::READY,100),request(2,2,State::PREPARING,0)},2000));
    auto second=f.snapshot({request(2,2,State::PREPARING,0)});ASSERT_EQ(1u,second.size());EXPECT_FALSE(second[0].totalMs);EXPECT_EQ(1u,f.model().samples());
    f.beginSession(10);EXPECT_EQ(2u,f.reservations());EXPECT_EQ(1u,f.model().samples());
}
TEST(MasterKVPreparationModelTest, PredictAndSmallSolveCostAreMeasured) {
    MasterKVPreparationModel m;for(int i=0;i<16;++i)ASSERT_TRUE(m.record({1,double(i%3),double(i%5),double(i%7)},100+i));
    volatile double sink=0;auto started=std::chrono::steady_clock::now();
    for(int i=0;i<100000;++i)sink+=*m.predict({1,2,3,4});
    auto ns=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-started).count();
    std::cout<<"kv_time_predict_ns="<<double(ns)/100000<<std::endl;
    started=std::chrono::steady_clock::now();for(int i=0;i<1000;++i)ASSERT_TRUE(m.record({1,double(i%3),double(i%5),double(i%7)},100+i%13));
    ns=std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-started).count();
    std::cout<<"kv_time_record_fit_ns="<<double(ns)/1000<<std::endl;EXPECT_GT(sink,0.0);
}
}
}

#include <cstdlib>
#include <fstream>
#include <sstream>
namespace navi_sched {
class MasterKVPreparationModelCorpusTest : public TESTBASE {};
TEST_F(MasterKVPreparationModelCorpusTest, ChronologicalFitsMatchIndependentNNLSObjective) {
    const auto overridePath=std::getenv("NAVI_KV_TIME_NNLS_CORPUS");
    const std::string path=overridePath ? overridePath : GET_PRIVATE_TEST_DATA_PATH()+"/kv_preparation/synthetic_nnls_reference.txt";
    std::ifstream input(path);ASSERT_TRUE(input.good());
    MasterKVPreparationModel model;
    std::vector<std::pair<MasterKVPreparationModel::Features,double>> history;
    std::string line;size_t rows=0,fullRankRows=0;double maximumRelativeLossDifference=0.0;
    while(std::getline(input,line)) {
        std::istringstream row(line);MasterKVPreparationModel::Features x,reference;
        for(auto &v:x)row>>v;double y,referenceLoss;int rank;row>>y>>referenceLoss>>rank;
        for(auto &v:reference)row>>v;ASSERT_FALSE(row.fail());
        ASSERT_TRUE(model.record(x,y));history.emplace_back(x,y);
        const auto coefficients=model.coefficients();long double loss=0.0;
        for(const auto &sample:history) {
            long double prediction=0.0;for(size_t i=0;i<4;++i)prediction+=coefficients[i]*sample.first[i];
            const auto error=prediction-sample.second;loss+=error*error;
        }
        const auto difference=std::abs(double(loss)-referenceLoss)/std::max(1.0,referenceLoss);
        maximumRelativeLossDifference=std::max(maximumRelativeLossDifference,difference);
        ASSERT_LE(difference,1e-7)<<"completed-load row "<<rows;
        for(auto c:coefficients){EXPECT_TRUE(std::isfinite(c));EXPECT_GE(c,0.0);}
        if(rank==4) {
            ++fullRankRows;
            for(size_t i=0;i<4;++i) EXPECT_NEAR(reference[i],coefficients[i],1e-5*std::max(1.0,std::abs(reference[i])));
        }
        ++rows;
    }
    EXPECT_EQ(overridePath ? 46u : 96u,rows);EXPECT_GT(fullRankRows,0u);
    std::cout<<"kv_time_nnls_relative_loss_difference="<<maximumRelativeLossDifference<<" full_rank_rows="<<fullRankRows<<std::endl;
}
}

namespace navi_sched {
TEST(MasterKVPreparationForecastsDelayedTest, ExecutionBeforeCompletionMetadataWaitsForFixedDuration) {
    MasterKVPreparationForecasts f;f.beginSession(10);ASSERT_TRUE(f.reserve(1,4096,0));
    NaviSchedKVPreparation r{1,1,1,1000,3000,true,NativeKVExecutionState::PREPARING};r.preparationElapsedMs=0;
    ASSERT_TRUE(f.observe(10,1,{r},1000));r.state=NativeKVExecutionState::EXECUTING;r.preparationElapsedMs=-1;
    ASSERT_TRUE(f.observe(10,2,{r},9000000));EXPECT_EQ(0u,f.model().samples());
    r.preparationElapsedMs=30;ASSERT_TRUE(f.observe(10,3,{r},12000000));EXPECT_EQ(1u,f.model().samples());
    EXPECT_NEAR(30.0,*f.model().predict({1,1,0,0}),1e-8);
    ASSERT_TRUE(f.observe(10,4,{r},15000000));EXPECT_EQ(1u,f.model().samples());
}
}

namespace navi_sched {
TEST(MasterKVPreparationForecastBoundaryTest, EveryElapsedAgeIsMonotoneAndPredictionBoundaryUsesFullLoad) {
    const MasterKVPreparationForecasts::Estimate frozen{1,2,1000000,0.0,100.0};
    double previous=-1.0;
    for(int64_t elapsedUs=0;elapsedUs<=200000;elapsedUs+=100) {
        const double weight=frozen.weight(1000000+elapsedUs,0);
        EXPECT_GE(weight,previous);EXPECT_GE(weight,0.0);EXPECT_LE(weight,1.0);
        EXPECT_NEAR(std::min(1.0,elapsedUs/100000.0),weight,1e-12);
        previous=weight;
    }
    EXPECT_DOUBLE_EQ(0.0,frozen.weight(1000000,0));
    EXPECT_LT(frozen.weight(1099999,0),1.0);
    EXPECT_DOUBLE_EQ(1.0,frozen.weight(1100000,0));
    EXPECT_DOUBLE_EQ(1.0,frozen.weight(5000000,0));
    EXPECT_DOUBLE_EQ(0.5,frozen.weight(1025000,25));
    EXPECT_DOUBLE_EQ(1.0,frozen.weight(1075000,25));
}
}

namespace navi_sched {
TEST(MasterKVPreparationForecastAuditTest, LogsOnlyNewLoadAndFixedOutcomeWithoutTrainingFromExecutionWait) {
    using State=NativeKVExecutionState;
    using Phase=MasterKVPreparationForecasts::AuditRecord::Phase;
    MasterKVPreparationForecasts forecast;forecast.beginSession(10);
    std::vector<MasterKVPreparationForecasts::AuditRecord> audit;
    ASSERT_TRUE(forecast.reserve(1,4096,0));
    NaviSchedKVPreparation load{1,1,1,100,3000,true,State::PREPARING};load.preparationElapsedMs=5;
    ASSERT_TRUE(forecast.observe(10,1,{load},1000,&audit));ASSERT_EQ(1u,audit.size());
    EXPECT_EQ(Phase::START,audit[0].phase);EXPECT_EQ(10u,audit[0].session);EXPECT_EQ(1u,audit[0].revision);
    EXPECT_EQ(1u,audit[0].estimate.requestId);EXPECT_EQ(1u,audit[0].estimate.generation);
    EXPECT_DOUBLE_EQ(5.0,audit[0].estimate.observedPreparationMs);EXPECT_FALSE(audit[0].estimate.totalMs);
    EXPECT_EQ(0u,audit[0].modelSamplesBefore);EXPECT_TRUE(audit[0].metadataKnown);EXPECT_TRUE(audit[0].observedPreparing);EXPECT_FALSE(audit[0].trained);
    EXPECT_EQ((MasterKVPreparationModel::Features{1,1,0,0}),audit[0].features);
    load.preparationElapsedMs=10;ASSERT_TRUE(forecast.observe(10,2,{load},6000,&audit));EXPECT_TRUE(audit.empty());
    load.state=State::EXECUTING;load.preparationElapsedMs=-1;
    ASSERT_TRUE(forecast.observe(10,3,{load},9000000,&audit));EXPECT_TRUE(audit.empty());EXPECT_EQ(0u,forecast.model().samples());
    load.preparationElapsedMs=100;
    ASSERT_TRUE(forecast.observe(10,4,{load},12000000,&audit));ASSERT_EQ(1u,audit.size());
    EXPECT_EQ(Phase::COMPLETE,audit[0].phase);EXPECT_DOUBLE_EQ(100.0,audit[0].actualPreparationMs);
    EXPECT_TRUE(audit[0].trained);EXPECT_EQ(0u,audit[0].modelSamplesBefore);EXPECT_EQ(1u,forecast.model().samples());
    ASSERT_TRUE(forecast.observe(10,5,{load},15000000,&audit));EXPECT_TRUE(audit.empty());
    ASSERT_TRUE(forecast.reserve(2,4096,0));load.requestId=2;load.loadGeneration=2;load.state=State::PREPARING;load.preparationElapsedMs=20;
    ASSERT_TRUE(forecast.observe(10,6,{load},16000000,&audit));ASSERT_EQ(1u,audit.size());
    EXPECT_EQ(Phase::START,audit[0].phase);ASSERT_TRUE(audit[0].estimate.totalMs);EXPECT_NEAR(100.0,*audit[0].estimate.totalMs,1e-8);
    EXPECT_EQ(1u,audit[0].modelSamplesBefore);
    load.state=State::FAILED;load.preparationElapsedMs=-1;
    ASSERT_TRUE(forecast.observe(10,7,{load},17000000,&audit));ASSERT_EQ(1u,audit.size());
    EXPECT_EQ(Phase::FAILED,audit[0].phase);EXPECT_FALSE(audit[0].trained);EXPECT_EQ(1u,forecast.model().samples());
    ASSERT_TRUE(forecast.observe(10,8,{load},18000000,&audit));EXPECT_TRUE(audit.empty());
    EXPECT_FALSE(forecast.observe(10,7,{load},19000000,&audit));EXPECT_TRUE(audit.empty());
    EXPECT_FALSE(forecast.observe(11,9,{load},19000000,&audit));EXPECT_TRUE(audit.empty());
}
TEST(MasterKVPreparationForecastAuditTest, MissingStartMetadataIsVisibleAndDoesNotFabricateTraining) {
    MasterKVPreparationForecasts forecast;forecast.beginSession(10);
    std::vector<MasterKVPreparationForecasts::AuditRecord> audit;
    NaviSchedKVPreparation load{3,1,3,100,3000,true,NativeKVExecutionState::READY};load.preparationElapsedMs=12;
    ASSERT_TRUE(forecast.observe(10,1,{load},1000,&audit));ASSERT_EQ(1u,audit.size());
    EXPECT_EQ(MasterKVPreparationForecasts::AuditRecord::Phase::COMPLETE,audit[0].phase);
    EXPECT_FALSE(audit[0].metadataKnown);EXPECT_FALSE(audit[0].observedPreparing);EXPECT_FALSE(audit[0].trained);
    EXPECT_FALSE(audit[0].estimate.totalMs);EXPECT_DOUBLE_EQ(12.0,audit[0].actualPreparationMs);
    EXPECT_EQ(0u,forecast.model().samples());
    ASSERT_TRUE(forecast.observe(10,2,{load},2000,&audit));EXPECT_TRUE(audit.empty());
    load.requestId=4;load.loadGeneration=4;load.state=NativeKVExecutionState::FAILED;load.preparationElapsedMs=-1;
    ASSERT_TRUE(forecast.observe(10,3,{load},3000,&audit));ASSERT_EQ(1u,audit.size());
    EXPECT_EQ(MasterKVPreparationForecasts::AuditRecord::Phase::FAILED,audit[0].phase);
    EXPECT_FALSE(audit[0].trained);EXPECT_EQ(0u,forecast.model().samples());
    ASSERT_TRUE(forecast.observe(10,4,{load},4000,&audit));EXPECT_TRUE(audit.empty());
}
}
