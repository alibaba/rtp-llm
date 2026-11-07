"""Invocation-local device checks, read once before permitting cache writes."""

import torch


class DeviceChecks:
    def __init__(self):
        self._checks = []
        self._flags = []
        self._offsets = []
        self.verified = False

    def add(self, invalid, message, error_type=RuntimeError):
        self.add_flags(invalid, [message], error_type)

    def add_flags(self, flags, messages, error_type=RuntimeError):
        self.verified = False
        self._offsets.append(len(self._checks))
        self._flags.append(flags)
        self._checks.extend((message, error_type) for message in messages)

    def verify(self, payload=None):
        self.verified = False
        if not self._checks:
            host_payload = None if payload is None else payload.tolist()
            self.verified = True
            return host_payload
        if len(self._checks) > 63:
            raise RuntimeError("too many QSA device checks in one invocation")
        if len(self._flags) == 1:
            verdict = self._flags[0].reshape(())
        elif self._flags[0].is_cuda and torch.version.hip is None:
            from .validation_triton import pack_checks

            verdict = pack_checks(self._flags, self._offsets)
        else:
            verdict = self._flags[0].to(torch.int64)
            for flags, shift in zip(self._flags[1:], self._offsets[1:]):
                verdict = verdict | (flags.to(torch.int64) << shift)
        if payload is None:
            host_verdict = int(verdict.item())
            host_payload = None
        else:
            packed = torch.cat((payload.reshape(-1).to(torch.int64), verdict[None]))
            host = packed.tolist()
            host_payload, host_verdict = host[:-1], host[-1]
        for bit, (message, error_type) in enumerate(self._checks):
            if message is not None and host_verdict & (1 << bit):
                raise error_type(message() if callable(message) else message)
        self.verified = True
        return host_payload
