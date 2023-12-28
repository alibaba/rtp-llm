import logging, traceback
from maga_transformer.aios.kmonitor.python_client.flume import ThriftSourceProtocol

from thrift.transport import TTransport, TSocket
from thrift.protocol import TCompactProtocol
logger = logging.getLogger('flume')


class _Transport(object):
    def __init__(self, thrift_host, thrift_port, timeout=None, unix_socket=None):
        self.thrift_host = thrift_host
        self.thrift_port = thrift_port
        self.timeout = timeout
        self.unix_socket = unix_socket

        self._socket = TSocket.TSocket(self.thrift_host, self.thrift_port, self.unix_socket)
        self._transport_factory = TTransport.TFramedTransportFactory()
        self._transport = self._transport_factory.getTransport(self._socket)

    def connect(self):
        try:
            if self.timeout:
                self._socket.setTimeout(self.timeout)
            if not self.is_open():
                self._transport = self._transport_factory.getTransport(self._socket)
                self._transport.open()
        except Exception as e:
            logger.warn('connect to flume exception:%s', e)
            logger.warn(traceback.format_exc())
            self.close()

    def reconnect(self):
        self.close()
        self.connect()

    def is_open(self):
        return self._transport.isOpen()

    def get_transport(self):
        return self._transport

    def close(self):
        self._transport.close()


class FlumeClient(object):
    def __init__(self, thrift_host, thrift_port, timeout=10000, unix_socket=None):
        self._transObj = _Transport(thrift_host, thrift_port, timeout=timeout, unix_socket=unix_socket)
        self._protocol = TCompactProtocol.TCompactProtocol(trans=self._transObj.get_transport())
        self.client = ThriftSourceProtocol.Client(iprot=self._protocol, oprot=self._protocol)
        self._transObj.connect()

    def send(self, event):
        try:
            self.client.send_append(event)
        except Exception as e:
            logger.warn('send to flume exception:%s', e)
            logger.warn(traceback.format_exc())
            self._transObj.reconnect()

    def send_batch(self, events):
        try:
            self.client.send_appendBatch(events)
        except Exception as e:
            logger.warn('send batch to flume exception:%s', e)
            logger.warn(traceback.format_exc())
            self._transObj.reconnect()

    def close(self):
        self._transObj.close()
