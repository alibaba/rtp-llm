"""Single-host EP consensus for the PPU Decode schedule."""

import mmap
import os
import tempfile

import torch.distributed as dist


def create_local_schedule_vote(group):
    from librtp_compute_ops import _PpuScheduleVote

    rank = dist.get_rank(group)
    ranks = dist.get_world_size(group)
    size = _PpuScheduleVote.size(ranks)
    owner = None
    if rank == 0:
        try:
            owner = tempfile.NamedTemporaryFile(prefix="rtp-ppu-schedule-", dir="/dev/shm")
            owner.truncate(size)
        except OSError:
            if owner is not None:
                owner.close()
            owner = None
    address = [None]
    if owner is not None:
        info = os.fstat(owner.fileno())
        address[0] = (owner.name, os.uname().nodename, info.st_dev, info.st_ino)
    dist.broadcast_object_list(address, group=group, group_src=0)
    if address[0] is None:
        return None

    path, host, device, inode = address[0]
    vote = None
    try:
        if os.uname().nodename == host:
            with open(path, "r+b") as source:
                info = os.fstat(source.fileno())
                if (info.st_dev, info.st_ino, info.st_size) == (device, inode, size):
                    mapping = mmap.mmap(source.fileno(), size)
                    vote = _PpuScheduleVote(mapping, rank, ranks)
    except OSError:
        pass
    ready = [False] * ranks
    try:
        dist.all_gather_object(ready, vote is not None, group=group)
    finally:
        if owner is not None:
            owner.close()
    return vote if all(ready) else None
