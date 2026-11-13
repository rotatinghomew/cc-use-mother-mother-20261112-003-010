"""Domain-specific errors with stable machine-readable codes."""

from __future__ import annotations


class LabError(Exception):
    code = "lab_error"

    def __init__(self, message: str, *, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def as_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "details": self.details}


class ContractError(LabError):
    code = "contract_error"


class IntegrityError(LabError):
    code = "integrity_error"


class CausalityError(LabError):
    code = "causality_error"


class CheckpointError(LabError):
    code = "checkpoint_error"


class ReplayError(LabError):
    code = "replay_error"


class PolicyError(LabError):
    code = "policy_error"


class ExperimentError(LabError):
    code = "experiment_error"


class VerificationError(LabError):
    code = "verification_error"

