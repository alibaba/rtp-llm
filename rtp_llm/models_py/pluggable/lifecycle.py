"""Public delayed initialization boundary, also called by PyWrappedModel.

The weak association records the actual factory binding. Omitting an optional
capability or replacing a model attribute cannot bypass context finalization.
The engine owns weights/cache/communication; failed contexts never free shared
resources and cannot be reused. Normal owner teardown releases the model.
"""

from weakref import WeakKeyDictionary

_bound_models = WeakKeyDictionary()


def bind_model_context(model, context):
    previous = _bound_models.get(model)
    if previous is not None and previous is not context:
        raise RuntimeError("Model is already bound to another construction context")
    _bound_models[model] = context


def initialize_model(model, init_resource):
    context = _bound_models.get(model)
    declared = getattr(model, "module_build_context", None)
    if context is None and declared is None:
        return model.initialize(init_resource)
    if context is None:
        raise RuntimeError("Model context has no factory root binding")
    try:
        entry_state = context.state
        if declared is not context or entry_state not in ("verified", "closed"):
            raise RuntimeError("Model initialize requires its verified factory context")
        if not context.weights_loaded:
            raise RuntimeError(
                "Model initialize requires the consumed weight preparation plan"
            )
        adapter = context.model_adapter
        if adapter is None or context.resource_plan is None:
            raise RuntimeError("Model initialize requires an adapter and resource plan")
        root_request = adapter.root_request(context.selection)
        # Warmup and the real executor bind different engine resources to the
        # same materialized model. Construction stays closed on later binds.
        if entry_state == "closed":
            context.validate_built_tree(model, root_path=root_request.path)
        adapter.validate_resources(model, init_resource, context)
        result = model.initialize(init_resource)
        if result is not True:
            raise RuntimeError("Model initialize did not succeed")
        if getattr(model, "module_build_context", None) is not context:
            raise RuntimeError("Model changed its context during initialize")
        if context.state != entry_state:
            raise RuntimeError(
                "Only the public initializer may finalize a model context"
            )
        context.validate_built_tree(model, root_path=root_request.path)
        adapter.validate_initialized_model(model, init_resource, context)
        binding = next(b for b in context.bindings if b.request == root_request)
        if binding.implementation.validate_initialized:
            from .spec import load_entrypoint

            load_entrypoint(binding.implementation.validate_initialized)(
                model, init_resource, context
            )
        if entry_state == "verified":
            context.close()
        return True
    except BaseException:
        context.fail()
        raise
