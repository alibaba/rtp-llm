"""Common owned launch boundary; discovery and record semantics are injected."""


def launch_client(operations, environment, directory, log_file, deadline, *, label=None):
    deadline.check()
    options = {} if label is None else {"label": label}
    process, output = operations.run_async(environment, directory, log_file, **options)
    try:
        deadline.check()
    except BaseException:
        from runtime.cleanup import stop_process
        stop_process(process, deadline)
        raise
    return process, output
