"""Run the existing physical-state oracles through HOST and DISK cache paths."""

import sys
import unittest
from typing import Final

from rtp_llm.test.smoke.deepseek_v4_tail_reuse_4layer_cp4_test import (
    DeepSeekV4TailReuseFourLayerCP4Test,
)
from rtp_llm.test.smoke.deepseek_v4_tail_reuse_4layer_cp_test import (
    DeepSeekV4TailReuseFourLayerCP2Test,
)
from rtp_llm.test.smoke.deepseek_v4_tail_reuse_4layer_mtp_cp_test import (
    DeepSeekV4TailReuseFourLayerMTPCP2Test,
)
from rtp_llm.test.smoke.deepseek_v4_tail_reuse_4layer_mtp_test import (
    DeepSeekV4TailReuseFourLayerMTPTest,
)
from rtp_llm.test.smoke.deepseek_v4_tail_reuse_4layer_test import (
    DeepSeekV4TailReuseFourLayerTest,
)
from rtp_llm.test.smoke.kimi_linear_common_prefix_test import KimiLinearCommonPrefixTest
from rtp_llm.test.smoke.local_tier_reuse import LocalCacheTier

_TIERS: Final[dict[str, LocalCacheTier]] = {"host": "host", "disk": "disk"}

if __name__ == "__main__":
    selected_tier = _TIERS[sys.argv.pop(1)]
    for fixture in (
        DeepSeekV4TailReuseFourLayerTest,
        DeepSeekV4TailReuseFourLayerCP2Test,
        DeepSeekV4TailReuseFourLayerCP4Test,
        DeepSeekV4TailReuseFourLayerMTPTest,
        DeepSeekV4TailReuseFourLayerMTPCP2Test,
        KimiLinearCommonPrefixTest,
    ):
        fixture.local_cache_tier = selected_tier
    unittest.main()
