class MinimalAlphaEvolveError(Exception):
    '''Base exception for the project.'''


class ConfigurationError(MinimalAlphaEvolveError):
    '''Raised when required configuration is missing or invalid.'''


class SaiaProtocolError(MinimalAlphaEvolveError):
    '''Raised when SAIA returns an unexpected response shape.'''


class CandidateValidationError(MinimalAlphaEvolveError):
    '''Raised when a candidate violates its representation contract.'''
