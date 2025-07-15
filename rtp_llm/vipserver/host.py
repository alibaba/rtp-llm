class Host:
    ip = None
    port = None

    def __init__(self, ip, port):
        self.ip = ip
        self.port = port

    def __str__(self):
        return "ip:%s,port:%d" % (self.ip, self.port)
