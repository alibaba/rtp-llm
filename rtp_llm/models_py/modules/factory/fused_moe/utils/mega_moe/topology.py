"""Validate the communicator against the expert weight partition."""


def validated_world_ep_group(config, dist):
    if not dist.is_initialized():
        raise RuntimeError("MegaMoE requires torch.distributed to be initialized")
    group = dist.group.WORLD
    size, rank = dist.get_world_size(group), dist.get_rank(group)
    if size != config.ep_size or rank != config.ep_rank:
        raise RuntimeError(
            "MegaMoE currently requires the EP group to equal WORLD: "
            f"runtime WORLD is rank {rank}/{size}, but "
            f"configuration EP is rank {config.ep_rank}/{config.ep_size}"
        )
    return group
