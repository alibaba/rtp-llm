"""DashScApp: standalone process hosting the DashSc gRPC server.

Parallel to ``FrontendApp`` but only serves the predict_v2 gRPC wire. Owns its own
asyncio event loop (the ``enqueue`` coroutine needs one) and its own
``BackendRPCServerVisitor`` so the two protocols can iterate independently.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import threading
import traceback
from typing import Any, List, Optional

from rtp_llm.config.log_config import get_log_path
from rtp_llm.config.py_config_modules import PyEnvConfigs
from rtp_llm.dash_sc.inference.servicer import (
    DashScInferenceServicer,
    build_think_runtime,
)
from rtp_llm.dash_sc.proxy.servicer import DashScProxyServicer
from rtp_llm.dash_sc.server import DashScGrpcServer
from rtp_llm.frontend.tokenizer_factory.tokenizer_factory import TokenizerFactory
from rtp_llm.metrics import kmonitor
from rtp_llm.model_factory import ModelFactory
from rtp_llm.openai.renderer_factory import ChatRendererFactory
from rtp_llm.openai.renderers.custom_renderer import RendererParams
from rtp_llm.server.backend_rpc_server_visitor import create_backend_rpc_server_visitor

# Env key that flips the process into reverse-proxy mode. Read once at
# ``start()`` entry so mode is decided at the process boundary — not re-probed
# inside the gRPC server or servicer. Empty / unset -> inference mode.
_FORWARD_ENV_KEY = "DASH_SC_GRPC_FORWARD_ADDR"


def _derive_echo_prefix_ids(generate_env_config: Any, base_tok: Any) -> List[int]:
    """Encode ``generate_env_config.think_start_tag`` once to produce the prefill token ids.

    Disabled (returns ``[]``) when ``THINK_MODE`` env is off or ``think_start_tag`` is empty;
    stays aligned with the engine's thinking switch so dash_sc and the engine turn on/off
    together. Fail-open: any error returns ``[]`` and logs a warning.
    """
    if not bool(getattr(generate_env_config, "think_mode", 0)):
        return []
    tag = getattr(generate_env_config, "think_start_tag", "") or ""
    if not tag:
        return []
    try:
        hf_tok = getattr(base_tok, "tokenizer", base_tok)
        ids = list(hf_tok.encode(tag, add_special_tokens=False))
    except Exception as e:
        logging.warning("[DashScApp] echo_prefix derive failed: %s", e)
        return []
    logging.info("[DashScApp] echo_prefix_ids=%s (think_start_tag=%r)", ids, tag)
    return ids


def _derive_stop_word_ids_list(
    model_config: Any, py_env_configs: PyEnvConfigs, base_tok: Any
) -> List[List[int]]:
    """Mirror ``openai_endpoint.__init__`` (rtp_llm/openai/openai_endpoint.py:75-150)
    stop-words assembly so the dash-sc gRPC path -- which bypasses the OpenAI endpoint
    because input is pre-tokenized upstream by dashscope-serving -- ends up with the
    same stop list. Sources merged in this order:
      1. model special_tokens (stop_words_id_list / stop_words_str_list);
      2. renderer-injected extras (e.g. GLM-5's <|user|>/<|observation|> via
         ChatGlm45Renderer._setup_stop_words);
      3. env-supplied stop_words_str / stop_words_list (with force_stop_words
         override, mirroring HTTP path semantics);
      4. str<->id bidirectional sync;
      5. dedup.
    Fail-open: any error returns [] and logs a warning so a renderer or env
    misconfiguration cannot break dash-sc startup.
    """
    try:
        gec = py_env_configs.generate_env_config
        special_tokens = model_config.special_tokens

        # Step 1: baseline from model config
        stop_words_id_list: List[List[int]] = [
            list(w) for w in (special_tokens.stop_words_id_list or [])
        ]
        stop_words_str_list: List[str] = list(special_tokens.stop_words_str_list or [])

        # Step 2: renderer extras
        params = RendererParams(
            model_type=model_config.model_type,
            max_seq_len=model_config.max_seq_len,
            eos_token_id=getattr(base_tok, "eos_token_id", None)
            or special_tokens.eos_token_id,
            stop_word_ids_list=list(stop_words_id_list),
            template_type=model_config.template_type,
            ckpt_path=model_config.ckpt_path,
        )
        renderer = ChatRendererFactory.get_renderer(
            base_tok,
            params,
            gec,
            py_env_configs.render_config,
            model_config.ckpt_path,
            getattr(py_env_configs, "misc_config", None),
            getattr(py_env_configs, "vit_config", None),
        )
        stop_words_id_list.extend(
            [list(w) for w in (renderer.get_all_extra_stop_word_ids_list() or [])]
        )

        # Step 3: env-supplied (mirror openai_endpoint.py:114-130 incl force_stop_words)
        env_str_list = json.loads(gec.stop_words_str) if gec.stop_words_str else []
        env_id_list = json.loads(gec.stop_words_list) if gec.stop_words_list else []
        if gec.force_stop_words:
            stop_words_str_list = list(env_str_list)
            stop_words_id_list = [list(w) for w in env_id_list]
        else:
            stop_words_str_list = stop_words_str_list + list(env_str_list)
            stop_words_id_list = stop_words_id_list + [list(w) for w in env_id_list]

        # Step 4: str<->id sync (mirror openai_endpoint.py:132-146)
        for ids in list(stop_words_id_list):
            try:
                w = base_tok.decode(ids)
                if w:
                    stop_words_str_list.append(w)
            except Exception:
                pass
        for s in list(stop_words_str_list):
            try:
                ids = base_tok.encode(s)
                if ids:
                    stop_words_id_list.append(list(ids))
            except Exception:
                pass

        # Step 5: dedup
        seen = set()
        out: List[List[int]] = []
        for ids in stop_words_id_list:
            t = tuple(ids)
            if t and t not in seen:
                seen.add(t)
                out.append(list(ids))
    except Exception as e:
        logging.warning("[DashScApp] stop_word derive failed: %s", e)
        return []
    logging.info("[DashScApp] derived stop_word_ids_list=%s", out)
    return out


class DashScApp:
    """Self-contained lifecycle for a per-rank DashSc gRPC server.

    Startup order (``start``):
      1. Pick mode from ``DASH_SC_GRPC_FORWARD_ADDR`` (proxy if set,
         inference otherwise). Inference mode additionally builds
         ``ModelConfig`` + ``BackendRPCServerVisitor``; proxy mode skips both
         since it only needs outbound channels to the configured backends.
      2. Construct the chosen servicer (``DashScProxyServicer`` or
         ``DashScInferenceServicer``).
      3. Spin up a dedicated asyncio loop in a background thread — same loop
         hosts the aio gRPC server AND backend ``enqueue`` coroutines, so the
         request path never leaves this loop.
      4. Call ``self._grpc_server.start_on_loop`` (schedules start on the
         loop and blocks the main thread until bind succeeds or raises).
      5. Notify the parent via the pipe, then block the main thread waiting on
         SIGTERM/SIGINT.
    """

    def __init__(self, py_env_configs: PyEnvConfigs):
        self.py_env_configs = py_env_configs
        self.server_config = py_env_configs.server_config
        self.dash_sc_grpc_config = py_env_configs.dash_sc_grpc_config

        self._enqueue_loop: Optional[asyncio.AbstractEventLoop] = None
        self._enqueue_loop_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()
        self._grpc_server = DashScGrpcServer(
            dash_sc_grpc_config=self.dash_sc_grpc_config
        )

    def _start_enqueue_loop(self) -> asyncio.AbstractEventLoop:
        loop = asyncio.new_event_loop()
        ready = threading.Event()

        def _run() -> None:
            asyncio.set_event_loop(loop)
            ready.set()
            loop.run_forever()

        thread = threading.Thread(target=_run, name="dash_sc_enqueue_loop", daemon=True)
        thread.start()
        if not ready.wait(timeout=5.0):
            raise RuntimeError("dash_sc enqueue asyncio loop failed to start")
        self._enqueue_loop = loop
        self._enqueue_loop_thread = thread
        return loop

    def _stop_enqueue_loop(self) -> None:
        loop = self._enqueue_loop
        if loop is None:
            return
        try:
            loop.call_soon_threadsafe(loop.stop)
        except RuntimeError:
            pass
        thread = self._enqueue_loop_thread
        if thread is not None:
            thread.join(timeout=5.0)
        try:
            loop.close()
        except Exception as e:
            logging.warning("[DashScApp] close enqueue loop failed: %s", e)
        self._enqueue_loop = None
        self._enqueue_loop_thread = None

    def _install_signal_handlers(self) -> None:
        def _handler(signum, frame):
            logging.info("[DashScApp] received signal %s, shutting down", signum)
            self._shutdown_event.set()

        try:
            signal.signal(signal.SIGTERM, _handler)
            signal.signal(signal.SIGINT, _handler)
        except ValueError:
            # signal handlers can only be installed on the main thread
            logging.warning(
                "[DashScApp] signal handlers not installed (not on main thread)"
            )

    def start(self, ready_pipe_writer=None) -> None:
        try:
            port = self.server_config.dash_sc_grpc_server_port
            is_proxy = bool(os.environ.get(_FORWARD_ENV_KEY, "").strip())

            if is_proxy:
                # Proxy mode: no model / weight loading / visitor needed — the
                # servicer is a transparent reverse proxy fed by outbound
                # channels to ``DASH_SC_GRPC_FORWARD_ADDR``. Skipping the
                # backend-visitor and model-config construction here avoids the
                # heavy engine init path (which would fail in proxy-only
                # deployments where no model is mounted).
                servicer: Any = DashScProxyServicer()
            else:
                model_config = ModelFactory.create_model_config(
                    model_args=self.py_env_configs.model_args,
                    lora_config=self.py_env_configs.lora_config,
                    kv_cache_config=self.py_env_configs.kv_cache_config,
                    profiling_debug_logging_config=self.py_env_configs.profiling_debug_logging_config,
                    generate_env_config=self.py_env_configs.generate_env_config,
                    embedding_config=self.py_env_configs.embedding_config,
                    quantization_config=self.py_env_configs.quantization_config,
                    render_config=self.py_env_configs.render_config,
                )

                backend_visitor = create_backend_rpc_server_visitor(
                    py_env_configs=self.py_env_configs,
                    model_config=model_config,
                )

                base_tok = TokenizerFactory.create(
                    model_config.ckpt_path,
                    model_config.tokenizer_path,
                    model_config.model_type,
                )
                echo_prefix_ids = _derive_echo_prefix_ids(
                    self.py_env_configs.generate_env_config, base_tok
                )
                extra_stop_word_ids = _derive_stop_word_ids_list(
                    model_config, self.py_env_configs, base_tok
                )
                # ``think_terminate_token_id`` <= 0 means the operator turned off
                # the in-stream "stop thinking" branch via env/args; carry that
                # through as ``None`` so the servicer skips the path entirely.
                env_terminate_id = (
                    self.py_env_configs.generate_env_config.think_terminate_token_id
                )
                think_runtime = build_think_runtime(
                    base_tok,
                    self.py_env_configs.generate_env_config,
                    model_config.model_type,
                    terminate_token_id=(
                        env_terminate_id if env_terminate_id > 0 else None
                    ),
                )
                servicer = DashScInferenceServicer(
                    backend_visitor=backend_visitor,
                    ip=self.server_config.ip,
                    port=port,
                    server_id=self.server_config.frontend_server_id,
                    echo_prefix_ids=echo_prefix_ids,
                    extra_stop_word_ids=extra_stop_word_ids,
                    tokenizer=base_tok,
                    generate_env_config=self.py_env_configs.generate_env_config,
                    think_runtime=think_runtime,
                )

            loop = self._start_enqueue_loop()

            # Register py_rtp_* metrics so the access-log interceptor's kmonitor.report
            # calls find their metric objects. Idempotent — matches FrontendServer.__init__
            # so dashboards/alerts see gRPC and HTTP paths under the same metric family
            # (split via the ``protocol`` tag the interceptor injects).
            kmonitor.init()

            logging.info(
                "[DashScApp] starting gRPC server rank_id=%s server_id=%s port=%s mode=%s",
                self.server_config.rank_id,
                self.server_config.frontend_server_id,
                port,
                "proxy" if is_proxy else "inference",
            )
            self._grpc_server.start_on_loop(
                loop,
                port=port,
                servicer=servicer,
                server_id=self.server_config.frontend_server_id,
                log_path=get_log_path(),
                backup_count=self.py_env_configs.profiling_debug_logging_config.log_file_backup_count,
                rank_id=self.server_config.rank_id,
            )
            logging.info("[DashScApp] gRPC server bound on port %s", port)
        except BaseException as e:
            error_trace = traceback.format_exc()
            logging.error("[DashScApp] start failed: %s\n%s", e, error_trace)
            if ready_pipe_writer is not None:
                try:
                    ready_pipe_writer.send(
                        {
                            "status": "failed",
                            "message": str(e),
                            "traceback": error_trace,
                        }
                    )
                    ready_pipe_writer.close()
                except Exception as pipe_error:
                    logging.warning(
                        "[DashScApp] failed to send failure via pipe: %s",
                        pipe_error,
                    )
            self._stop_enqueue_loop()
            raise

        if ready_pipe_writer is not None:
            try:
                ready_pipe_writer.send(
                    {
                        "status": "success",
                        "message": (
                            f"DashSc gRPC server started on rank "
                            f"{self.server_config.rank_id} "
                            f"server {self.server_config.frontend_server_id}"
                        ),
                    }
                )
                ready_pipe_writer.close()
            except Exception as e:
                logging.warning("[DashScApp] failed to send success via pipe: %s", e)

        self._install_signal_handlers()
        logging.info("[DashScApp] entering service loop (SIGTERM/SIGINT to stop)")
        try:
            self._shutdown_event.wait()
        finally:
            self.stop()

    def stop(self) -> None:
        to = self.server_config.shutdown_timeout
        grace = None if to < 0 else float(to)
        try:
            self._grpc_server.stop(grace)
        except Exception as e:
            logging.warning("[DashScApp] grpc_server.stop failed: %s", e)
        self._stop_enqueue_loop()
