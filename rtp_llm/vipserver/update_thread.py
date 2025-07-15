import threading
import time


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
            self.func()
