class MinimalAlphaEvolveError(Exception):
    """Base exception for the project."""


class ConfigurationError(MinimalAlphaEvolveError):
    """Raised when required configuration is missing or invalid."""


class SaiaProtocolError(MinimalAlphaEvolveError):
    """Raised when SAIA returns an unexpected response shape."""


class CandidateValidationError(MinimalAlphaEvolveError):
    """Raised when a candidate violates its representation contract."""


class ArchiveError(MinimalAlphaEvolveError):
    """Raised when an archive cannot be read or updated safely."""


class ArchiveFormatError(ArchiveError):
    """Raised when an archive contains a malformed record."""


class DuplicateCandidateError(ArchiveError):
    """Raised when a candidate ID conflicts with an archived candidate."""


class DuplicateEvaluationError(ArchiveError):
    """Raised when an evaluation conflicts with an archived evaluation."""
