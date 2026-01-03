from rtp_llm.frontend.generation.orchestrator import GenerationOrchestrator

# Backwards compatibility: old imports expecting Pipeline now get the new orchestrator.
Pipeline = GenerationOrchestrator

__all__ = ["GenerationOrchestrator", "Pipeline"]
