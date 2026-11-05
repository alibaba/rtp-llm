"""Roll back frontend registrations when NaviSched dispatch cannot complete."""

import asyncio
import logging
from contextvars import ContextVar
from functools import wraps

_LOGGER = logging.getLogger(__name__)
_TRANSACTION = ContextVar("navi_sched_request_registration", default=None)
_MARKER = "_navi_sched_registration_rollback"


class _RegistrationTransaction:
    def __init__(self, owner):
        self.owner = owner
        self.task = asyncio.current_task()
        self.active = True
        self.states = {}
        self.parents = {}

    def record(self, processor, request_id, state, parent, prior_parent):
        if (
            not self.active
            or asyncio.current_task() is not self.task
            or processor is not self.owner.output_processor
        ):
            return
        self.states[request_id] = state
        if parent is not None and prior_parent is None:
            self.parents.setdefault(parent.request_id, parent)

    def _can_abort_parent(self, processor, parent):
        return all(
            (state := processor.request_states.get(request_id)) is None
            or self.states.get(request_id) is state
            for request_id in parent.child_requests
        )

    async def rollback(self):
        # Finish local cleanup before awaiting backend cancellation. Even another
        # cancellation must not leave the large prompt in request_states.
        self.active = False
        processor = self.owner.output_processor
        backend_ids = []
        for request_id, parent in self.parents.items():
            if processor.parent_requests.get(
                request_id
            ) is parent and self._can_abort_parent(processor, parent):
                backend_ids.extend(
                    processor.abort_requests([request_id], internal=True)
                )
        request_ids = [
            request_id
            for request_id, state in self.states.items()
            if processor.request_states.get(request_id) is state
        ]
        if request_ids:
            backend_ids.extend(processor.abort_requests(request_ids, internal=True))
        unique_backend_ids = list(dict.fromkeys(backend_ids))
        if unique_backend_ids:
            await self.owner.engine_core.abort_requests_async(unique_backend_ids)
        remaining = sum(
            processor.request_states.get(request_id) is state
            for request_id, state in self.states.items()
        )
        _LOGGER.info(
            "[navi-frontend-registration-rollback] registered=%d backend_aborted=%d remaining=%d",
            len(self.states),
            len(unique_backend_ids),
            remaining,
        )

    def close(self):
        # Background tasks can inherit a ContextVar. Leave those tasks an inert,
        # empty transaction rather than retaining completed requests through it.
        self.active = False
        self.states.clear()
        self.parents.clear()
        self.owner = None
        self.task = None


async def _registered_call(owner, original, args, kwargs):
    existing = _TRANSACTION.get()
    if (
        existing is not None
        and existing.active
        and existing.owner is owner
        and existing.task is asyncio.current_task()
    ):
        return await original(owner, *args, **kwargs)
    transaction = _RegistrationTransaction(owner)
    token = _TRANSACTION.set(transaction)
    try:
        return await original(owner, *args, **kwargs)
    except BaseException:
        try:
            await transaction.rollback()
        except BaseException as cleanup_error:
            # Keep the original rejection/timeout/cancellation and its status.
            _LOGGER.error(
                "NaviSched frontend registration rollback failed: cleanup_error_type=%s",
                type(cleanup_error).__name__,
            )
        raise
    finally:
        transaction.close()
        _TRANSACTION.reset(token)


def install_request_registration_rollback(async_llm_type, output_processor_type):
    installed = async_llm_type.__dict__.get(_MARKER)
    if installed is not None:
        if installed is not output_processor_type:
            raise RuntimeError(
                "NaviSched registration guard has a different output processor"
            )
        return
    if _MARKER in output_processor_type.__dict__:
        raise RuntimeError(
            "NaviSched output processor already belongs to another registration guard"
        )

    original_add = async_llm_type.add_request
    original_inner_add = async_llm_type._add_request
    original_register = output_processor_type.add_request

    @wraps(original_add)
    async def add_request(self, *args, **kwargs):
        return await _registered_call(self, original_add, args, kwargs)

    @wraps(original_inner_add)
    async def inner_add_request(self, *args, **kwargs):
        return await _registered_call(self, original_inner_add, args, kwargs)

    @wraps(original_register)
    def register(self, request, prompt, parent_req=None, request_index=0, queue=None):
        transaction = _TRANSACTION.get()
        if (
            transaction is None
            or not transaction.active
            or transaction.task is not asyncio.current_task()
            or self is not transaction.owner.output_processor
        ):
            return original_register(
                self, request, prompt, parent_req, request_index, queue
            )
        prior = self.request_states.get(request.request_id)
        prior_parent = (
            self.parent_requests.get(parent_req.request_id)
            if parent_req is not None
            else None
        )
        succeeded = False
        try:
            result = original_register(
                self, request, prompt, parent_req, request_index, queue
            )
            succeeded = True
            return result
        finally:
            state = self.request_states.get(request.request_id)
            if state is not None and (succeeded or state is not prior):
                transaction.record(
                    self, request.request_id, state, parent_req, prior_parent
                )

    output_processor_type.add_request = register
    async_llm_type.add_request = add_request
    async_llm_type._add_request = inner_add_request
    setattr(async_llm_type, _MARKER, output_processor_type)
    setattr(output_processor_type, _MARKER, async_llm_type)

    _LOGGER.info("[navi-frontend-registration-guard] installed")
