"""
app/llm/exceptions.py

Provider-agnostic LLM exception hierarchy.

These exceptions are raised by the service layer and caught by the API layer.
No exception class in this file should reference any specific provider SDK —
provider implementations must catch SDK-specific errors and re-raise as
LLMProviderError.
"""


class LLMError(Exception):
    """
    Base exception for all LLM-related errors.
    Catch this to handle any LLM failure generically.
    """


class LLMProviderError(LLMError):
    """
    Raised when the underlying LLM provider returns an error or is
    unreachable. The original SDK exception should be chained via `raise
    LLMProviderError(...) from original_exc` to preserve the full traceback
    while keeping the application layer decoupled from SDK internals.
    """


class LLMConfigurationError(LLMError):
    """
    Raised at startup if the LLM provider cannot be configured — for example,
    if an unknown provider name is specified or a required API key is missing.
    """
