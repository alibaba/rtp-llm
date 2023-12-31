#
# Autogenerated by Thrift Compiler (0.10.0)
#
# DO NOT EDIT UNLESS YOU ARE SURE THAT YOU KNOW WHAT YOU ARE DOING
#
#  options string: py
#

from thrift.Thrift import TType, TMessageType, TFrozenDict, TException, TApplicationException
from thrift.protocol.TProtocol import TProtocolException
import sys

from thrift.transport import TTransport


class Status(object):
    OK = 0
    FAILED = 1
    ERROR = 2
    UNKNOWN = 3

    _VALUES_TO_NAMES = {
        0: "OK",
        1: "FAILED",
        2: "ERROR",
        3: "UNKNOWN",
    }

    _NAMES_TO_VALUES = {
        "OK": 0,
        "FAILED": 1,
        "ERROR": 2,
        "UNKNOWN": 3,
    }


class ThriftFlumeEvent(object):
    """
    Attributes:
     - headers
     - body
    """

    thrift_spec = (
        None,  # 0
        (1, TType.MAP, 'headers', (TType.STRING, 'UTF8', TType.STRING, 'UTF8', False), None, ),  # 1
        (2, TType.STRING, 'body', 'BINARY', None, ),  # 2
    )

    def __init__(self, headers=None, body=None,):
        self.headers = headers
        self.body = body

    def read(self, iprot):
        if iprot._fast_decode is not None and isinstance(iprot.trans, TTransport.CReadableTransport) and self.thrift_spec is not None:
            iprot._fast_decode(self, iprot, (self.__class__, self.thrift_spec))
            return
        iprot.readStructBegin()
        while True:
            (fname, ftype, fid) = iprot.readFieldBegin()
            if ftype == TType.STOP:
                break
            if fid == 1:
                if ftype == TType.MAP:
                    self.headers = {}
                    (_ktype1, _vtype2, _size0) = iprot.readMapBegin()
                    for _i4 in range(_size0):
                        _key5 = iprot.readString().decode('utf-8') if sys.version_info[0] == 2 else iprot.readString()
                        _val6 = iprot.readString().decode('utf-8') if sys.version_info[0] == 2 else iprot.readString()
                        self.headers[_key5] = _val6
                    iprot.readMapEnd()
                else:
                    iprot.skip(ftype)
            elif fid == 2:
                if ftype == TType.STRING:
                    self.body = iprot.readBinary()
                else:
                    iprot.skip(ftype)
            else:
                iprot.skip(ftype)
            iprot.readFieldEnd()
        iprot.readStructEnd()

    def write(self, oprot):
        if oprot._fast_encode is not None and self.thrift_spec is not None:
            oprot.trans.write(oprot._fast_encode(self, (self.__class__, self.thrift_spec)))
            return
        oprot.writeStructBegin('ThriftFlumeEvent')
        if self.headers is not None:
            oprot.writeFieldBegin('headers', TType.MAP, 1)
            oprot.writeMapBegin(TType.STRING, TType.STRING, len(self.headers))
            for kiter7, viter8 in list(self.headers.items()):
                oprot.writeString(kiter7.encode('utf-8') if sys.version_info[0] == 2 else kiter7)
                oprot.writeString(viter8.encode('utf-8') if sys.version_info[0] == 2 else viter8)
            oprot.writeMapEnd()
            oprot.writeFieldEnd()
        if self.body is not None:
            oprot.writeFieldBegin('body', TType.STRING, 2)
            oprot.writeBinary(self.body)
            oprot.writeFieldEnd()
        oprot.writeFieldStop()
        oprot.writeStructEnd()

    def validate(self):
        if self.headers is None:
            raise TProtocolException(message='Required field headers is unset!')
        if self.body is None:
            raise TProtocolException(message='Required field body is unset!')
        return

    def __repr__(self):
        L = ['%s=%r' % (key, value)
             for key, value in list(self.__dict__.items())]
        return '%s(%s)' % (self.__class__.__name__, ', '.join(L))

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.__dict__ == other.__dict__

    def __ne__(self, other):
        return not (self == other)
