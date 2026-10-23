"""Continue a bounded request on its existing executor, without waiting there."""

from concurrent.futures import Future


def continue_after(source, executor, finish, cancel, on_error):
    """Keep a terminal Future for admission/ownership, not an occupied thread.

    The caller retains its admission slot until the returned future completes.
    No executor or unbounded independent work queue is created here.
    """
    terminal = Future()

    def run():
        if not terminal.set_running_or_notify_cancel():
            return
        try:
            terminal.set_result(finish(source.result()))
        except BaseException as error:
            try:
                on_error(error)
            finally:
                terminal.set_exception(error)
        finally:
            cancel()

    def ready(_):
        if terminal.cancelled():
            return
        try:
            queued = executor.submit(run)
            # shutdown(cancel_futures=True) can discard a queued continuation.
            # Propagate that cancellation to its admission/ownership future.
            queued.add_done_callback(
                lambda done: terminal.cancel() if done.cancelled() else None
            )
        except Exception as error:
            if terminal.set_running_or_notify_cancel():
                try:
                    on_error(error)
                finally:
                    terminal.set_exception(error)
                    cancel()

    def cancelled(done):
        if done.cancelled():
            cancel()

    terminal.add_done_callback(cancelled)
    source.add_done_callback(ready)
    return terminal
