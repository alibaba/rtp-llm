"""Expert placement for multi-node tensor parallelism with a single EP rank."""

import json
from typing import List, Optional


def pure_tp_identity_placement(
    layer_num: int,
    expert_num: int,
    phy_exp_num: int,
    num_nodes: int,
    phy2log_path: Optional[str] = None,
) -> List[List[int]]:
    """Keep every expert on every TP rank; physical hosts do not partition EP.

    This branch deliberately supports nonredundant EP=1 only. The normal
    single-node and EP>1 placement algorithms stay in LoadConfig unchanged.
    """
    for name, value in (
        ("layer_num", layer_num),
        ("expert_num", expert_num),
        ("phy_exp_num", phy_exp_num),
        ("num_nodes", num_nodes),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"pure TP {name} must be a positive integer")
    if num_nodes <= 1:
        raise ValueError("multi-node pure TP requires num_nodes > 1")
    if phy_exp_num != expert_num:
        raise ValueError("multi-node EP=1 requires nonredundant experts")
    identity = list(range(expert_num))
    if phy2log_path:
        with open(phy2log_path) as source:
            placement = json.load(source)
        if (
            not isinstance(placement, list)
            or len(placement) != layer_num
            or any(
                not isinstance(row, list)
                or any(type(expert) is not int for expert in row)
                or row != identity
                for row in placement
            )
        ):
            raise ValueError("multi-node pure TP requires a complete identity mapping in every layer")
        return placement
    # Independent rows are required because load balancing updates them in place.
    return [identity.copy() for _ in range(layer_num)]
