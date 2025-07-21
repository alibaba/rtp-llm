from enum import Enum
from typing import Dict

import torch
from libth_transformer.rtp_llm_ops import RtpProcessGroup, RtpProcessGroupType


class Group(Enum):
    DP = "DP"
    TP = "TP"
    DP_AND_TP = "DP_AND_TP"


_group_map: Dict[Group, RtpProcessGroup] = {}


def _group_type_convert(group: Group) -> RtpProcessGroupType:
    if group == Group.DP:
        return RtpProcessGroupType.DP_GROUP
    elif group == Group.TP:
        return RtpProcessGroupType.TP_GROUP
    elif group == Group.DP_AND_TP:
        return RtpProcessGroupType.DP_AND_TP_GROUP
    raise ValueError(f"Invalid group: {group}")


def _get_group(group: Group) -> RtpProcessGroup:
    if group not in _group_map:
        _group_map[group] = RtpProcessGroup(_group_type_convert(group))
    return _group_map[group]


def send(tensor: torch.Tensor, dst: int, group: Group = Group.DP_AND_TP) -> None:
    _get_group(group).send([tensor], dst)


def recv(
    tensor: torch.Tensor, src: int, group: Group = Group.DP_AND_TP
) -> torch.Tensor:
    return _get_group(group).recv([tensor], src)


def broadcast(tensor: torch.Tensor, src: int, group: Group = Group.DP_AND_TP) -> None:
    _get_group(group).broadcast([tensor], src)


def all_reduce(tensor: torch.Tensor, group: Group = Group.DP_AND_TP) -> torch.Tensor:
    return _get_group(group).all_reduce([tensor])


def all_gather(tensor: torch.Tensor, group: Group = Group.DP_AND_TP) -> torch.Tensor:
    raise NotImplementedError("AllGather is not implemented")
