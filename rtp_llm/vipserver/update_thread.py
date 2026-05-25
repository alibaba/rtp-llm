import logging
import threading
import time
import traceback


class UpdateThread(threading.Thread):
    def __init__(self, name, func, interval=10):
        threading.Thread.__init__(self)
        self.name = name
        self.func = func
        self.interval = interval
        self.stop_flag = False
        self.setDaemon(True)

    def run(self):
        while not self.stop_flag:
            time.sleep(self.interval)
            try:
                self.func()
            except Exception as e:
                stack_summary = traceback.format_exception(type(e), e, e.__traceback__)
                logging.error(
                    f"vipserver update thread {self.name} caught exception, continuing loop: {e}\n"
                    + "\n".join(stack_summary)
                )
