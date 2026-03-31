class ThresholdConfig:
    # Similarity thresholds (cosine, 0-1)
    MERGE_NODE          = 0.72
    DUPLICATE_MERGE     = 0.88
    WEAK_EDGE           = 0.58
    QUESTION_DEDUP_HIGH = 0.90
    QUESTION_DEDUP_LOW  = 0.70
    COHERENCE           = 0.65
    GAP_CONFIDENCE      = 0.75
    GAP_DEDUP           = 0.75
    CONTRADICTION       = 0.60
    AGENDA_PREFILTER    = 0.30


THRESHOLDS = ThresholdConfig()


class ModelConfig:
    """
    Per-role model selection.

    Override any of these to use different models for different tasks:
        MODELS.CREATIVE = "llama3.1:70b"   # bigger model for dreaming
        MODELS.PRECISE  = "qwen2.5:7b"     # faster model for JSON extraction
    """
    # Creative tasks: dreaming, synthesis, analogies
    CREATIVE     = "mixtral:latest"
    # Precise tasks: JSON extraction, factual answers
    PRECISE      = "mixtral:latest"
    # Code generation: sandbox experiments
    CODE         = "mixtral:latest"
    # Deliberate reasoning: thinker, chain-of-thought
    REASONING    = "mixtral:latest"
    # Conversation: chat interface
    CONVERSATION = "mixtral:latest"


MODELS = ModelConfig()

