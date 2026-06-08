"""DeepSeek-V4 MoE routed-expert strategies.

Each strategy implements the ``RoutedExpertsStrategy`` interface from
``base.py``. ``select_strategy`` picks one based on (ep_size, env-vars,
kernel availability).

Importing this package populates the strategy registry (each
``@register_strategy``-decorated class registers itself on import). Order
of import below = priority for ``select_strategy(forced=None)`` auto-pick.
EP>1 is special-cased to require Mega and fail fast when it is unavailable.
"""

from .base import (
    MoeCfg,
    RoutedExpertsStrategy,
    select_strategy,
)

# Side-effect import — each module's ``@register_strategy`` decorator pushes
# the class into the priority list as the import lands. Order here = priority
# (high→low):
try:
    from .mega import MegaMoEStrategy        # noqa: F401  ep_size>1 + SM100 + dist
except ImportError:
    pass
try:
    from .grouped_fp4 import (               # noqa: F401  ep_size==1 + kernel
        GroupedFP4Strategy,
        _has_fp8_fp4_grouped_kernel,
    )
except (ImportError, AttributeError):
    _has_fp8_fp4_grouped_kernel = lambda: False
try:
    from .deepep import DeepEPStrategy       # noqa: F401  ep_size>1 fallback
except ImportError:
    pass
from .local_loop import LocalLoopStrategy  # noqa: F401  universal fallback

__all__ = [
    "MoeCfg",
    "RoutedExpertsStrategy",
    "select_strategy",
    "_has_fp8_fp4_grouped_kernel",
]
