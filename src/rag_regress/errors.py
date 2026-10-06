"""Application-specific exception types."""


class UserInputError(Exception):
    """Raised when a user-supplied file or value is invalid."""


class ExecutionError(Exception):
    """Raised when a requested operation cannot be completed."""
