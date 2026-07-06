import unittest

from rtp_llm.dash_sc.inference.servicer import DashScInferenceServicer
from rtp_llm.dash_sc.proto import predict_v2_pb2
from rtp_llm.dash_sc.proxy.servicer import DashScProxyServicer


class _HealthState:
    def __init__(self) -> None:
        self.unavailable = False

    def is_unavailable(self) -> bool:
        return self.unavailable


class DashScHealthTest(unittest.IsolatedAsyncioTestCase):
    async def test_health_contract_for_both_servicers(self) -> None:
        for servicer_type in (DashScInferenceServicer, DashScProxyServicer):
            with self.subTest(servicer_type=servicer_type.__name__):
                state = _HealthState()
                servicer = servicer_type(health_state=state)

                live = await servicer.ServerLive(
                    predict_v2_pb2.ServerLiveRequest(), None
                )
                ready = await servicer.ServerReady(
                    predict_v2_pb2.ServerReadyRequest(), None
                )
                model_ready = await servicer.ModelReady(
                    predict_v2_pb2.ModelReadyRequest(), None
                )
                self.assertTrue(live.live)
                self.assertTrue(ready.ready)
                self.assertTrue(model_ready.ready)

                state.unavailable = True
                ready = await servicer.ServerReady(
                    predict_v2_pb2.ServerReadyRequest(), None
                )
                model_ready = await servicer.ModelReady(
                    predict_v2_pb2.ModelReadyRequest(), None
                )
                live = await servicer.ServerLive(
                    predict_v2_pb2.ServerLiveRequest(), None
                )
                self.assertFalse(ready.ready)
                self.assertFalse(model_ready.ready)
                self.assertTrue(live.live)


if __name__ == "__main__":
    unittest.main()
