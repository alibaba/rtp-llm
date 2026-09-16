"""Validate the process group used by fused expert-parallel kernels."""


def get_validated_world_ep_group(cfg, dist):
    if not dist.is_initialized():
        raise RuntimeError("MegaMoE requires torch.distributed to be initialized")
    group = dist.group.WORLD
    actual_size = dist.get_world_size(group)
    actual_rank = dist.get_rank(group)
    if actual_size != cfg.ep_size or actual_rank != cfg.ep_rank:
        raise RuntimeError(
            "MegaMoE currently requires the EP group to equal WORLD: "
            f"runtime WORLD is rank {actual_rank}/{actual_size}, but "
            f"configuration EP is rank {cfg.ep_rank}/{cfg.ep_size}"
        )
    return group
