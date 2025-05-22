import time
import logging
from datetime import timedelta
from torch.distributed.rendezvous import rendezvous

def create_store(master_url, world_rank, world_size):
    rendezvous_iterator = rendezvous(
        master_url, world_rank, world_size
    )
    store, rank, world_size = next(rendezvous_iterator)        
    logging.info(f"store: {store}, rank: {rank}, world_size: {world_size}")
    store.set_timeout(timedelta(seconds=10))          
    return store

def store_based_barrier(rank: int, world_size: int, store, timeout: timedelta):
    store_key = "{}:{}".format("store_barrier", 0)
    store.add(store_key, 1)
    logging.info("Added key: {} to store for rank: {}".format(store_key, rank))

    # Now wait for all workers to check in with the store.
    # Use 'add' instead of 'get' since for some store implementations 'add'
    # doesn't work well with 'get'. Ideally the store implementations should
    # be fixed, but for backward compatiblity reasons it is risky to change
    # the store implementations. Once, we completely migrate away from these
    # legacy stores, we can use 'get' here instead.
    worker_count = store.add(store_key, 0)
    start = time.time()
    log_time = time.time()
    while worker_count != world_size:
        time.sleep(0.01)
        worker_count = store.add(store_key, 0)

        # Print status periodically to keep track.
        if timedelta(seconds=(time.time() - log_time)) > timedelta(seconds=10):
            logging.info(
                "Waiting in store based barrier to initialize process group for "
                "rank: {}, key: {} (world_size={}, worker_count={}, timeout={})".format(
                    rank, store_key, world_size, worker_count, timeout
                )
            )
            log_time = time.time()

        if timedelta(seconds=(time.time() - start)) > timeout:
            raise RuntimeError(
                "Timed out initializing process group in store based barrier on "
                "rank: {}, for key: {} (world_size={}, worker_count={}, timeout={})".format(
                    rank, store_key, world_size, worker_count, timeout
                )
            )

    logging.info(
        f"Rank {rank}: Completed store-based barrier for key:{store_key} with {world_size} nodes."
    )