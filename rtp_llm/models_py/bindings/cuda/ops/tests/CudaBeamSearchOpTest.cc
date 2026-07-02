#include "rtp_llm/cpp/testing/BeamSearchOpTest.hpp"

using namespace std;
using namespace rtp_llm;

class CudaBeamSearchOpTest: public BeamSearchOpTest {};

TEST_F(CudaBeamSearchOpTest, simpleTest) {
    runSimpleTests();
}

TEST_F(CudaBeamSearchOpTest, variableBeamWidthTest) {
    runVariableBeamWidthTests();
}
