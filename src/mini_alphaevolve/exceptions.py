class MinimalAlphaEvolveError(Exception):
    """Base exception for the project."""


class ConfigurationError(MinimalAlphaEvolveError):
    """Raised when required configuration is missing or invalid."""


class SaiaProtocolError(MinimalAlphaEvolveError):
    """Raised when SAIA returns an unexpected response shape."""


class SaiaTransientError(MinimalAlphaEvolveError):
    """Raised for a SAIA transport failure that is safe to retry."""


class CandidateValidationError(MinimalAlphaEvolveError):
    """Raised when a candidate violates its representation contract."""


class CandidateEvaluationError(MinimalAlphaEvolveError):
    """Raised when a valid candidate cannot produce a finite scalar result."""


class ArchiveError(MinimalAlphaEvolveError):
    """Raised when an archive cannot be read or updated safely."""


class ArchiveFormatError(ArchiveError):
    """Raised when an archive contains a malformed record."""


class DuplicateCandidateError(ArchiveError):
    """Raised when a candidate ID conflicts with an archived candidate."""


class DuplicateEvaluationError(ArchiveError):
    """Raised when an evaluation conflicts with an archived evaluation."""
