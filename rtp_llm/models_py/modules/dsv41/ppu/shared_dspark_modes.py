"""Startup and row contracts for shared PPU DSpark; no sampling policy here.

The public executor owns budget allocation, packed eligibility, Markov/RNG,
acceptance and bonus tokens. A scalar budget is a batch bank, never a bound
on an individual request's verified width. Dense fallback and cancellation
therefore require gamma+1 rollback rows even for packed budget0.
"""

import json
import os
from dataclasses import asdict, dataclass

CONFIG_KEY = "_ppu_shared_dspark_mode"
PROFILE_ENV = "RTP_V41_PPU_DSPARK_PROFILE"
ANCHOR_ENV = "RTP_V41_PPU_DSPARK_SAMPLE_FROM_ANCHOR"


@dataclass(frozen=True)
class SharedDSparkMode:
    gamma: int
    confidence: str = "off"
    verify: str = "dense"
    budget: int = -1
    budget_mode: str = "fixed"
    sample_from_anchor: bool = True

    def validate(self):
        if type(self.gamma) is not int or self.gamma not in (1, 3, 5):
            raise ValueError("shared PPU DSpark supports gamma1/3/5")
        if self.confidence not in ("off", "observe", "bounded"):
            raise ValueError("invalid shared confidence mode")
        if self.verify not in ("dense", "packed") or self.budget_mode not in ("fixed", "adaptive"):
            raise ValueError("invalid shared verification/budget mode")
        if type(self.budget) is not int or not -1 <= self.budget <= self.gamma:
            raise ValueError("shared budget must be -1 or in [0,gamma]")
        if type(self.sample_from_anchor) is not bool:
            raise ValueError("sample_from_anchor must be boolean")
        if not self.sample_from_anchor and self.confidence != "off":
            raise ValueError("public confidence requires sampling from anchor")
        if (self.budget != -1 or self.verify == "packed" or self.budget_mode == "adaptive") and self.confidence != "bounded":
            raise ValueError("public budget/packed verification requires bounded confidence")
        if self.verify == "packed" and self.budget < 0:
            raise ValueError("public packed verification requires an explicit budget")
        if self.budget_mode == "adaptive" and self.verify != "packed":
            raise ValueError("public adaptive budget requires packed verification")
        return self

    @property
    def query_width(self):
        return self.gamma + int(not self.sample_from_anchor)

    @property
    def max_verified_rows(self):
        return self.gamma + 1

    @classmethod
    def from_config(cls, config=None, env=None):
        config = config or {}
        env = os.environ if env is None else env
        frozen = config.get(CONFIG_KEY)
        if frozen is None and env.get(PROFILE_ENV):
            frozen = json.loads(env[PROFILE_ENV])
        if frozen is not None:
            return cls(**frozen).validate()
        if env.get("SP_TYPE", "").lower() != "dspark":
            return None
        return cls(
            gamma=int(env.get("GEN_NUM_PER_CIRCLE", "0")),
            confidence=env.get("SP_DSPARK_CONFIDENCE_MODE", "off"),
            verify=env.get("SP_DSPARK_VERIFY_MODE", "dense"),
            budget=int(env.get("SP_DSPARK_VERIFY_BUDGET", "-1")),
            budget_mode=env.get("SP_DSPARK_VERIFY_BUDGET_MODE", "fixed"),
            sample_from_anchor=_anchor_flag(env.get(ANCHOR_ENV, "1")),
        ).validate()

    def freeze(self, config):
        return dict(config, **{CONFIG_KEY: asdict(self)})

    def protocol(self):
        return {
            **asdict(self), "proposal_query_width": self.query_width,
            "maximum_verified_input_rows": self.max_verified_rows,
            "minimum_accepted_input_rows": 0,
            "maximum_backtracking_tokens": self.max_verified_rows,
            "budget_owner": "public-executor-batch-bank",
            "state_layout": "native-accepted-row-journal-v2",
        }


def target_loop_geometry(lengths, packed):
    """View only: PPU Attention uses actual request-major state.rows().

    The inherited layer loop needs a rectangular hidden tensor, not uniform
    logical requests. A [1,total_rows] view keeps every real packed row in its
    original order without padding, a fabricated average or extra collectives.
    """
    if not lengths or any(n <= 0 for n in lengths):
        raise ValueError("target verification requires nonempty request groups")
    if packed:
        return 1, sum(lengths)
    if len(set(lengths)) != 1:
        raise ValueError("dense target verification requires uniform lengths")
    return len(lengths), lengths[0]


def _anchor_flag(value):
    if str(value).lower() in ("1", "true", "yes", "on"):
        return True
    if str(value).lower() in ("0", "false", "no", "off"):
        return False
    raise ValueError("invalid PPU DSpark anchor flag")
