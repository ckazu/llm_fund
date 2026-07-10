"""Enumerations shared across the domain layer.

Values are lowercase/uppercase strings (not plain ints) so that they read
directly in SQLite rows, JSON payloads and audit logs without a lookup table.
"""

from enum import StrEnum


class Action(StrEnum):
    """Trading action requested by an order plan / instruction."""

    BUY = "BUY"
    SELL = "SELL"
    CLOSE = "CLOSE"


class InstructionStatus(StrEnum):
    """Lifecycle status of a validated instruction (instructions.status)."""

    PENDING = "pending"
    FILLED = "filled"
    PARTIAL = "partial"
    SKIPPED = "skipped"
    EXPIRED = "expired"
    REJECTED = "rejected"


class ExecutionStatus(StrEnum):
    """Outcome of a human-recorded execution (executions.status)."""

    FILLED = "filled"
    PARTIAL = "partial"
    SKIPPED = "skipped"


class FillExitReason(StrEnum):
    """Reason a virtual fill's position was closed (virtual_fills.exit_reason)."""

    TP = "tp"
    SL = "sl"
    EXPIRY = "expiry"
    MANUAL = "manual"


class ProposalStatus(StrEnum):
    """Approval lifecycle for policies/criteria proposals."""

    DRAFT = "draft"
    ACTIVE = "active"
    SUPERSEDED = "superseded"


class PendingOrderStatus(StrEnum):
    """Virtual IFO order lifecycle tracked in `pending_orders`.

    Independent of `InstructionStatus` (which tracks the *human* execution): this
    is the conservative virtual-fill engine's own view of the limit order — still
    waiting (pending), filled, or expired unfilled.
    """

    PENDING = "pending"
    FILLED = "filled"
    EXPIRED = "expired"
