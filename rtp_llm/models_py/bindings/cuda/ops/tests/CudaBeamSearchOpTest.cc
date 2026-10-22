#include "rtp_llm/cpp/testing/BeamSearchOpTest.hpp"

using namespace std;
using namespace rtp_llm;

class CudaBeamSearchOpTest: public BeamSearchOpTest {};

TEST_F(CudaBeamSearchOpTest, rejectsUnsupportedBeamWidths) {
    runUnsupportedBeamWidthTests();
}

TEST_F(CudaBeamSearchOpTest, simpleTest) {
    runSimpleTests();
}

TEST_F(CudaBeamSearchOpTest, variableBeamWidthTest) {
    runVariableBeamWidthTests();
}

TEST_F(CudaBeamSearchOpTest, prunedVocabV2Test) {
    runPrunedVocabV2Tests();
}

TEST_F(CudaBeamSearchOpTest, forcedV2HandlesMaskedEqualSmallBeams) {
    runMaskedEqualSmallBeamV2Tests();
}

TEST_F(CudaBeamSearchOpTest, selectAllVocabV2Test) {
    runSelectAllVocabV2Tests();
}
