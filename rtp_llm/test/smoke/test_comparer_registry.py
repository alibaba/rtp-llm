import pytest

from rtp_llm.test.smoke import comparer_registry


class DecodeArpcComparer:
    pass


class EmbeddingArpcComparer:
    pass


class MainseComparer:
    pass


@pytest.fixture(autouse=True)
def restore_registry():
    original_registry = list(comparer_registry._REGISTRY)
    original_fallback = comparer_registry._FALLBACK
    comparer_registry._reset_for_tests()
    yield
    comparer_registry._REGISTRY[:] = original_registry
    comparer_registry._FALLBACK = original_fallback


def _register_mainse_comparers() -> None:
    comparer_registry._register_mainse_comparer_classes(
        DecodeArpcComparer,
        EmbeddingArpcComparer,
        MainseComparer,
    )


@pytest.mark.parametrize("mainse_flag", ["mainse_module", "mainse"])
def test_plain_mainse_uses_http_comparer(mainse_flag: str):
    _register_mainse_comparers()

    assert (
        comparer_registry.resolve_comparer({mainse_flag: True}, "/")
        is MainseComparer
    )


def test_decode_arpc_takes_priority_over_plain_mainse():
    _register_mainse_comparers()

    assert (
        comparer_registry.resolve_comparer(
            {"mainse_module": True, "use_decode_arpc": True}, "/"
        )
        is DecodeArpcComparer
    )


def test_embedding_arpc_takes_priority_over_plain_mainse():
    _register_mainse_comparers()

    assert (
        comparer_registry.resolve_comparer(
            {"mainse_module": True, "use_emb_arpc": True}, "/"
        )
        is EmbeddingArpcComparer
    )
