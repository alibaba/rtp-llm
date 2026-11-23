import threading


class UpdateThread(threading.Thread):
    def __init__(self, name, func, interval=10):
        threading.Thread.__init__(self)
        self.name = name
        self.func = func
        self.interval = interval
        self._stop_event = threading.Event()
        self.setDaemon(True)

    @property
    def stop_flag(self):
        return self._stop_event.is_set()

    @stop_flag.setter
    def stop_flag(self, value):
        if value:
            self._stop_event.set()
        else:
            self._stop_event.clear()

    def run(self):
        while not self._stop_event.wait(self.interval):
            self.func()
