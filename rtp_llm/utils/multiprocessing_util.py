import multiprocessing.util
import os
import shutil
import tempfile


def ensure_short_ipc_temp_dir() -> str:
    directory = multiprocessing.util.get_temp_dir()
    listener_path = os.path.join(directory, "listener-xxxxxxxx")
    if len(os.fsencode(listener_path)) <= 107:
        return directory
    directory = tempfile.mkdtemp(prefix="pymp-", dir="/tmp")
    multiprocessing.util.Finalize(
        None, shutil.rmtree, args=(directory,), exitpriority=-100
    )
    multiprocessing.current_process()._config["tempdir"] = directory
    return directory
