"""Custom exceptions for easycluster, giving beginner-friendly error messages."""


class EasyClusterError(Exception):
    """Base class for all easycluster errors."""


class InputValidationError(EasyClusterError):
    """Raised when the input DataFrame or feature selection is invalid."""


class NotFittedError(EasyClusterError):
    """Raised when a method requires a fitted model but none exists yet."""


class ParameterError(EasyClusterError):
    """Raised when an invalid or incompatible parameter combination is used."""
