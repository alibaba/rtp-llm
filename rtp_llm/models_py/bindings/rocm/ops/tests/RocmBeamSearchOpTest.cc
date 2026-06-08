#include "rtp_llm/cpp/testing/BeamSearchOpTest.hpp"

using namespace std;
using namespace rtp_llm;

class RocmBeamSearchOpTest: public BeamSearchOpTest {};

TEST_F(RocmBeamSearchOpTest, simpleTest) {
    runSimpleTests();
}

TEST_F(RocmBeamSearchOpTest, variableBeamWidthTest) {
    runVariableBeamWidthTests();
}
