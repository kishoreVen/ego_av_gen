
"""Exponential backoff retry logic for handling transient failures."""

import logging
from dataclasses import dataclass

from tenacity import (
    Retrying,
    before_sleep_log,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

logger = logging.getLogger(__name__)


def is_retryable_error(error: Exception) -> bool:
    """Determine if an error should trigger a retry.

    Args:
        error: The exception that was raised

    Returns:
        True if the error is retryable, False otherwise
    """
    # Never retry signals or user interrupts — let them propagate immediately.
    if isinstance(error, (KeyboardInterrupt, SystemExit)):
        return False

    # Check for transient API failures that should be retried
    error_msg = str(error).lower()
    transient_value_errors = [
        "no images found",  # Gemini sometimes returns empty image responses
        "no text found",    # Similar transient failures
    ]
    if isinstance(error, ValueError):
        if any(msg in error_msg for msg in transient_value_errors):
            return True
        # These are programming errors, not transient failures
        if any(msg in error_msg for msg in [
            "query empty",
            "not supported",
            "invalid",
            "must be",
            "requires",
        ]):
            return False

    # Check for specific error types that indicate transient failures
    error_msg = str(error).lower()
    retryable_patterns = [
        "connection",
        "timeout",
        "rate limit",
        "too many requests",
        "429",
        "500",
        "502",
        "503",
        "504",
        "temporary",
        "unavailable",
        "network",
        "socket",
        "disconnected",
    ]

    if any(pattern in error_msg for pattern in retryable_patterns):
        return True

    # Check error type name for connection/timeout errors
    error_type = type(error).__name__.lower()
    if any(pattern in error_type for pattern in ["connection", "timeout", "http"]):
        return True

    # For generic exceptions, be conservative and retry
    # (but not for ValueError which we already handled)
    if not isinstance(error, ValueError):
        return True

    return False


@dataclass
class RetryConfig:
    """Configuration for exponential backoff retry logic."""
    max_retries: int = 3
    base_delay: float = 1.0  # seconds
    max_delay: float = 32.0  # seconds
    exponential_base: float = 2.0
    jitter_factor: float = 0.25  # 0-25% of base delay

    def attempts(self, operation: str) -> Retrying:
        """Return a tenacity Retrying iterator for exponential backoff."""
        return Retrying(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential_jitter(
                initial=self.base_delay,
                max=self.max_delay,
                exp_base=self.exponential_base,
                jitter=self.base_delay * self.jitter_factor,
            ),
            retry=retry_if_exception(is_retryable_error),
            reraise=True,
            before_sleep=before_sleep_log(logger, logging.WARNING),
        )
