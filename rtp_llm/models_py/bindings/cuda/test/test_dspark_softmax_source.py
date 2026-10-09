"""CPU-only provenance, routing, and DSpark sampling scope checks."""
import hashlib
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[5]
KERNELS = ROOT / "rtp_llm/models_py/bindings/cuda/kernels"


class DSparkSoftmaxSourceTest(unittest.TestCase):
    def test_pinned_payload(self):
        source = KERNELS / "blackwell_softmax_vendor/cake_blackwell_softmax_cached_cluster.cu"
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),
                         "c7f6ab44832d655e92ff5ebad4b22c9c0cebbcc9ae8776cfa9952f56a219156b")

    def test_policy_boundaries_and_coverage(self):
        code = r'''
#include "rtp_llm/models_py/bindings/cuda/kernels/speculative_sampling/dspark_softmax_policy.h"
#include <cassert>
using namespace rtp_llm::dspark_softmax;
int main() {
  const auto p = reinterpret_cast<void*>(32);
  assert(useCached(5, 196608, 10, 0, p, p));
  assert(useCached(5, 196608, 10, 3, p, p));
  assert(!useCached(5, 196608, 9, 0, p, p));
  assert(!useCached(5, 196608, 10, 1, p, p));
  assert(!useCached(0, 196608, 10, 0, p, p));
  assert(!useCached(5, 196607, 10, 0, p, p));
  assert(!useCached(5, 262152, 10, 0, p, p));
  assert(!useCached(5, 196608, 10, 0, reinterpret_cast<void*>(4), p));
  assert(!useCached(129, 32000, 10, 0, p, p));
  assert(useCached(128, 32000, 10, 0, p, p));
  assert(useCached(385, 32000, 10, 0, p, p));
  assert(!useCached(2147483647LL, 8, 10, 0, p, p));
  for (int rows : {1, 3, 5, 8, 16, 128, 385, 1024}) {
    for (int vocab : {8, 256, 8192, 32000, 65536, 196608, 262144}) {
      auto v = selectVariant(rows, vocab, 148);
      assert(v.cluster_ctas > 0 && v.cluster_ctas <= 16);
      assert(v.max_tiles == 4 || v.max_tiles == 8);
      assert(chunkElements(vocab, v.cluster_ctas) <= v.max_tiles * kTileElements);
      assert(chunkElements(vocab, v.cluster_ctas) * v.cluster_ctas >= vocab);
    }
  }
}
'''
        # Compile and execute a temporary binary without any GPU access.
        import tempfile
        with tempfile.TemporaryDirectory(prefix="dspark-softmax-policy-") as tmp:
            binary = str(Path(tmp) / "policy")
            subprocess.run(["g++", "-std=c++17", "-I", str(ROOT), "-x", "c++", "-", "-o", binary],
                           input=code, text=True, check=True, capture_output=True)
            subprocess.run([binary], check=True)

    def test_sampling_scope_and_rng(self):
        source = (ROOT / "rtp_llm/cpp/normal_engine/speculative/SpeculativeSampler.cc").read_text()
        draft = source.split("SamplerOutput SpeculativeSampler::sampleDSparkDraft", 1)[1].split(
            "torch::Tensor SpeculativeSampler::computeDSparkConfidence", 1)[0]
        self.assertIn("execDSparkSoftmax(logits, softmax_workspace)", draft)
        self.assertIn("execDSparkCombineLogits", draft)
        self.assertIn("execSampleFromProbs(sampling_probabilities)", draft)
        self.assertIn("execReserveSampleFromProbsRng", draft)
        self.assertEqual(source.count("execDSparkSoftmax("), 1)
        self.assertIn("output.all_probs = torch::softmax(logits, -1)", source)
        native = (KERNELS / "speculative_sampling/dspark_softmax.cu").read_text()
        self.assertIn("OnlineSoftmax<float>", native)
        self.assertNotIn("pybind", native)
        self.assertNotIn("static torch::Tensor", native)


if __name__ == "__main__":
    unittest.main()
