"""Typed error taxonomy for Nexus (spec §36)."""


class NexusError(Exception):
    """Base class for all application-level Nexus errors."""


class InvalidInputError(NexusError):
    pass


class InvalidCustomerIdError(NexusError):
    pass


class InvalidIdentifierError(NexusError):
    pass


class IntakeValidationError(NexusError):
    pass


class PromptInjectionError(NexusError):
    pass


class RetrievalError(NexusError):
    pass


class RetrievalConflictError(NexusError):
    pass


class MCPServerUnavailableError(NexusError):
    pass


class MCPToolError(NexusError):
    pass


class UnauthorizedFieldAccessError(NexusError):
    pass


class ResourceValidationError(NexusError):
    pass


class DraftGroundingError(NexusError):
    pass


class ValidationError(NexusError):
    pass


class EscalationError(NexusError):
    pass


class NotificationError(NexusError):
    pass


class AuditLogError(NexusError):
    pass


class RateLimitedError(NexusError):
    """Too many requests for one customer id, or overall, within the rate-limit window."""
