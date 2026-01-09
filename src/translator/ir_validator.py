"""IR Validator - checks referential integrity of StrategyIR.

Ensures all references to indicators, states, and other entities
actually exist in the IR. This catches bugs where translation
produces IR that references non-existent entities.
"""

from dataclasses import dataclass, field

from .ir import (
    AllOfCondition,
    AnyOfCondition,
    CompareCondition,
    Condition,
    ExpressionValue,
    IndicatorBandValue,
    IndicatorPropertyValue,
    IndicatorValue,
    NotCondition,
    RegimeCondition,
    StateValue,
    StrategyIR,
)


@dataclass
class ValidationError:
    """A single validation error."""

    path: str  # Where in the IR the error occurred
    message: str  # What's wrong


@dataclass
class ValidationResult:
    """Result of validating an IR."""

    errors: list[ValidationError] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def add_error(self, path: str, message: str) -> None:
        self.errors.append(ValidationError(path=path, message=message))


class IRValidator:
    """Validates StrategyIR for internal consistency."""

    def __init__(self, ir: StrategyIR):
        self.ir = ir
        self.result = ValidationResult()

        # Build lookup sets for defined entities
        self.indicator_ids: set[str] = {ind.id for ind in ir.indicators}
        self.state_ids: set[str] = {state.id for state in ir.state}

    def validate(self) -> ValidationResult:
        """Run all validations and return result."""
        # Validate entry rule
        if self.ir.entry:
            self._validate_condition(self.ir.entry.condition, "entry.condition")
            self._validate_action(self.ir.entry.action, "entry.action")

        # Validate exit rules
        for i, exit_rule in enumerate(self.ir.exits):
            self._validate_condition(exit_rule.condition, f"exits[{i}].condition")
            self._validate_action(exit_rule.action, f"exits[{i}].action")

        # Validate gates
        for i, gate in enumerate(self.ir.gates):
            self._validate_condition(gate.condition, f"gates[{i}].condition")

        # Validate on_bar hooks
        for i, op in enumerate(self.ir.on_bar):
            self._validate_state_op(op, f"on_bar[{i}]")

        # Validate on_bar_invested hooks
        for i, op in enumerate(self.ir.on_bar_invested):
            self._validate_state_op(op, f"on_bar_invested[{i}]")

        return self.result

    def _validate_condition(self, cond: Condition | None, path: str) -> None:
        """Validate a condition and its nested values."""
        if cond is None:
            return

        if isinstance(cond, CompareCondition):
            self._validate_value(cond.left, f"{path}.left")
            self._validate_value(cond.right, f"{path}.right")
        elif isinstance(cond, (AllOfCondition, AnyOfCondition)):
            for i, sub_cond in enumerate(cond.conditions):
                self._validate_condition(sub_cond, f"{path}.conditions[{i}]")
        elif isinstance(cond, NotCondition):
            self._validate_condition(cond.condition, f"{path}.condition")
        elif isinstance(cond, RegimeCondition):
            # RegimeCondition uses lookback_bars, not indicator refs
            pass
        # Other condition types don't have value references

    def _validate_value(self, value, path: str) -> None:
        """Validate a value reference."""
        if value is None:
            return

        if isinstance(value, IndicatorValue):
            if value.indicator_id not in self.indicator_ids:
                self.result.add_error(
                    path,
                    f"References undefined indicator '{value.indicator_id}'. "
                    f"Defined indicators: {sorted(self.indicator_ids)}",
                )
        elif isinstance(value, IndicatorBandValue):
            if value.indicator_id not in self.indicator_ids:
                self.result.add_error(
                    path,
                    f"References undefined indicator '{value.indicator_id}'. "
                    f"Defined indicators: {sorted(self.indicator_ids)}",
                )
        elif isinstance(value, IndicatorPropertyValue):
            if value.indicator_id not in self.indicator_ids:
                self.result.add_error(
                    path,
                    f"References undefined indicator '{value.indicator_id}'. "
                    f"Defined indicators: {sorted(self.indicator_ids)}",
                )
        elif isinstance(value, StateValue):
            if value.state_id not in self.state_ids:
                self.result.add_error(
                    path,
                    f"References undefined state '{value.state_id}'. "
                    f"Defined states: {sorted(self.state_ids)}",
                )
        elif isinstance(value, ExpressionValue):
            self._validate_value(value.left, f"{path}.left")
            self._validate_value(value.right, f"{path}.right")
        # PriceValue, VolumeValue, TimeValue, LiteralValue don't reference entities

    def _validate_state_op(self, op, path: str) -> None:
        """Validate a state operation."""
        if hasattr(op, "state_id"):
            if op.state_id not in self.state_ids:
                self.result.add_error(
                    path,
                    f"References undefined state '{op.state_id}'. "
                    f"Defined states: {sorted(self.state_ids)}",
                )
        if hasattr(op, "value"):
            self._validate_value(op.value, f"{path}.value")
        if hasattr(op, "condition"):
            self._validate_condition(op.condition, f"{path}.condition")

    def _validate_action(self, action, path: str) -> None:
        """Validate an action."""
        # Currently actions don't have entity references
        # But we can extend this if they do in the future
        pass


def validate_ir(ir: StrategyIR) -> ValidationResult:
    """Convenience function to validate an IR."""
    return IRValidator(ir).validate()
