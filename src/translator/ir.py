"""Intermediate Representation (IR) for strategy translation.

This module defines the typed IR that sits between MCP card schemas and LEAN runtime.
Uses Pydantic for serialization and discriminated unions for polymorphism.

The IR is:
- Fully typed (no magic strings, no dict access)
- Serializable to/from JSON
- Self-evaluating (conditions know how to evaluate themselves)
- Self-executing (actions know how to execute themselves)
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, Field

# =============================================================================
# Enums
# =============================================================================


class Resolution(str, Enum):
    """Trading resolution/timeframe."""

    MINUTE = "Minute"
    HOUR = "Hour"
    DAILY = "Daily"


class CompareOp(str, Enum):
    """Comparison operators."""

    LT = "<"
    LTE = "<="
    GT = ">"
    GTE = ">="
    EQ = "=="
    NEQ = "!="

    def apply(self, left: float, right: float) -> bool:
        """Apply the comparison operator."""
        match self:
            case CompareOp.LT:
                return left < right
            case CompareOp.LTE:
                return left <= right
            case CompareOp.GT:
                return left > right
            case CompareOp.GTE:
                return left >= right
            case CompareOp.EQ:
                return left == right
            case CompareOp.NEQ:
                return left != right


class PriceField(str, Enum):
    """Price bar fields."""

    OPEN = "open"
    HIGH = "high"
    LOW = "low"
    CLOSE = "close"


class BandField(str, Enum):
    """Band indicator fields (for BB, KC, etc.)."""

    UPPER = "upper"
    MIDDLE = "middle"
    LOWER = "lower"


class IndicatorProperty(str, Enum):
    """Indicator properties beyond the main value."""

    STANDARD_DEVIATION = "StandardDeviation"
    BAND_WIDTH = "BandWidth"


class StateType(str, Enum):
    """State variable types."""

    FLOAT = "float"
    INT = "int"
    BOOL = "bool"


# =============================================================================
# Indicators
# =============================================================================


class EMA(BaseModel):
    """Exponential Moving Average."""

    type: Literal["EMA"] = "EMA"
    id: str
    period: int


class SMA(BaseModel):
    """Simple Moving Average."""

    type: Literal["SMA"] = "SMA"
    id: str
    period: int


class BollingerBands(BaseModel):
    """Bollinger Bands."""

    type: Literal["BB"] = "BB"
    id: str
    period: int
    multiplier: float = 2.0


class KeltnerChannel(BaseModel):
    """Keltner Channel."""

    type: Literal["KC"] = "KC"
    id: str
    period: int
    multiplier: float = 2.0


class ATR(BaseModel):
    """Average True Range."""

    type: Literal["ATR"] = "ATR"
    id: str
    period: int


class Maximum(BaseModel):
    """Maximum (highest high) over period."""

    type: Literal["MAX"] = "MAX"
    id: str
    period: int


class Minimum(BaseModel):
    """Minimum (lowest low) over period."""

    type: Literal["MIN"] = "MIN"
    id: str
    period: int


class RateOfChange(BaseModel):
    """Rate of Change (momentum)."""

    type: Literal["ROC"] = "ROC"
    id: str
    period: int


class ADX(BaseModel):
    """Average Directional Index."""

    type: Literal["ADX"] = "ADX"
    id: str
    period: int


class DonchianChannel(BaseModel):
    """Donchian Channel (highest high / lowest low over period)."""

    type: Literal["DC"] = "DC"
    id: str
    period: int


class VWAP(BaseModel):
    """Volume Weighted Average Price."""

    type: Literal["VWAP"] = "VWAP"
    id: str
    period: int = 0  # 0 = intraday (resets daily), >0 = rolling period


class RollingWindow(BaseModel):
    """Rolling window for tracking historical values (e.g., previous close for gaps)."""

    type: Literal["RW"] = "RW"
    id: str
    period: int
    field: PriceField = PriceField.CLOSE


class VolumeSMA(BaseModel):
    """Simple Moving Average of volume."""

    type: Literal["VOL_SMA"] = "VOL_SMA"
    id: str
    period: int


class RollingMinMax(BaseModel):
    """Rolling min/max tracker for recent highs/lows."""

    type: Literal["RMM"] = "RMM"
    id: str
    period: int
    mode: Literal["min", "max"]
    field: PriceField = PriceField.CLOSE


# Discriminated union of all indicator types
Indicator = Annotated[
    EMA
    | SMA
    | BollingerBands
    | KeltnerChannel
    | ATR
    | Maximum
    | Minimum
    | RateOfChange
    | ADX
    | DonchianChannel
    | VWAP
    | RollingWindow
    | VolumeSMA
    | RollingMinMax,
    Field(discriminator="type"),
]


# =============================================================================
# Value References (things that resolve to a float)
# =============================================================================


class IndicatorValue(BaseModel):
    """Reference to an indicator's current value."""

    type: Literal["indicator"] = "indicator"
    indicator_id: str


class IndicatorBandValue(BaseModel):
    """Reference to a band indicator's specific band (upper/middle/lower)."""

    type: Literal["indicator_band"] = "indicator_band"
    indicator_id: str
    band: BandField


class IndicatorPropertyValue(BaseModel):
    """Reference to an indicator's property (e.g., StandardDeviation, BandWidth)."""

    type: Literal["indicator_property"] = "indicator_property"
    indicator_id: str
    property: IndicatorProperty


class PriceValue(BaseModel):
    """Reference to current price bar field."""

    type: Literal["price"] = "price"
    field: PriceField = PriceField.CLOSE


class VolumeValue(BaseModel):
    """Reference to current bar's volume."""

    type: Literal["volume"] = "volume"


class TimeValue(BaseModel):
    """Reference to current bar's time component."""

    type: Literal["time"] = "time"
    component: Literal["hour", "minute", "day_of_week"]


class StateValue(BaseModel):
    """Reference to a state variable."""

    type: Literal["state"] = "state"
    state_id: str


class LiteralValue(BaseModel):
    """A literal numeric value."""

    type: Literal["literal"] = "literal"
    value: float


class ExpressionValue(BaseModel):
    """A computed expression (left op right)."""

    type: Literal["expr"] = "expr"
    op: Literal["+", "-", "*", "/"]
    left: ValueRef
    right: ValueRef


# Discriminated union of all value reference types
ValueRef = Annotated[
    IndicatorValue
    | IndicatorBandValue
    | IndicatorPropertyValue
    | PriceValue
    | VolumeValue
    | TimeValue
    | StateValue
    | LiteralValue
    | ExpressionValue,
    Field(discriminator="type"),
]

# Update forward refs for ExpressionValue
ExpressionValue.model_rebuild()


# =============================================================================
# Conditions (things that evaluate to bool)
# =============================================================================


class CompareCondition(BaseModel):
    """Compare two values."""

    type: Literal["compare"] = "compare"
    left: ValueRef
    op: CompareOp
    right: ValueRef


class AllOfCondition(BaseModel):
    """All conditions must be true (AND)."""

    type: Literal["allOf"] = "allOf"
    conditions: list[Condition]


class AnyOfCondition(BaseModel):
    """Any condition must be true (OR)."""

    type: Literal["anyOf"] = "anyOf"
    conditions: list[Condition]


class NotCondition(BaseModel):
    """Negate a condition."""

    type: Literal["not"] = "not"
    condition: Condition


class RegimeCondition(BaseModel):
    """A regime-based condition (maps to ConditionSpec.regime from MCP schema).

    This is a higher-level condition that the evaluator expands based on metric type.
    """

    type: Literal["regime"] = "regime"
    metric: str  # e.g., "trend_ma_relation", "ret_pct", "vol_regime"
    op: CompareOp
    value: float | str
    # Metric-specific params
    ma_fast: int | None = None
    ma_slow: int | None = None
    lookback_bars: int | None = None


class BreakoutCondition(BaseModel):
    """Breakout condition - price breaks above N-bar high or below N-bar low.

    Evaluator checks direction and compares price to max/min indicator.
    """

    type: Literal["breakout"] = "breakout"
    lookback_bars: int = 50
    buffer_bps: int = 0
    max_indicator: str  # ID of MAX indicator
    min_indicator: str  # ID of MIN indicator


class SqueezeCondition(BaseModel):
    """Squeeze condition - volatility compression followed by expansion.

    Evaluator checks if BB width percentile is below threshold.
    """

    type: Literal["squeeze"] = "squeeze"
    squeeze_metric: str = "bb_width_pctile"  # Metric for compression detection
    pctile_threshold: float = 10.0  # Compression percentile threshold
    break_rule: str = "donchian"  # How to detect breakout
    with_trend: bool = False  # Whether to require trend alignment


class TimeFilterCondition(BaseModel):
    """Time-based filter condition.

    Evaluator checks current time against allowed days/windows.
    """

    type: Literal["time_filter"] = "time_filter"
    days_of_week: list[int] = []  # 0=Mon, 6=Sun
    time_window: str = ""  # e.g., "09:30-16:00"
    days_of_month: list[int] = []
    timezone: str = "UTC"


# Discriminated union of all condition types
Condition = Annotated[
    CompareCondition
    | AllOfCondition
    | AnyOfCondition
    | NotCondition
    | RegimeCondition
    | BreakoutCondition
    | SqueezeCondition
    | TimeFilterCondition,
    Field(discriminator="type"),
]

# Update forward refs for recursive types
AllOfCondition.model_rebuild()
AnyOfCondition.model_rebuild()
NotCondition.model_rebuild()


# =============================================================================
# Actions
# =============================================================================


class SetHoldingsAction(BaseModel):
    """Set portfolio holdings to a target allocation."""

    type: Literal["set_holdings"] = "set_holdings"
    allocation: float = 0.95


class LiquidateAction(BaseModel):
    """Liquidate all holdings."""

    type: Literal["liquidate"] = "liquidate"


class MarketOrderAction(BaseModel):
    """Place a market order for specific quantity."""

    type: Literal["market_order"] = "market_order"
    quantity: float


# Discriminated union of all action types
Action = Annotated[
    SetHoldingsAction | LiquidateAction | MarketOrderAction,
    Field(discriminator="type"),
]


# =============================================================================
# State Variables
# =============================================================================


class StateVar(BaseModel):
    """A state variable declaration."""

    id: str
    var_type: StateType = StateType.FLOAT
    default: float | int | bool | None = None


# =============================================================================
# State Mutations (for on_fill, on_bar hooks)
# =============================================================================


class SetStateOp(BaseModel):
    """Set a state variable to a value."""

    type: Literal["set_state"] = "set_state"
    state_id: str
    value: ValueRef


class IncrementStateOp(BaseModel):
    """Increment a state variable by 1."""

    type: Literal["increment"] = "increment"
    state_id: str


class MaxStateOp(BaseModel):
    """Set state to max of current value and new value."""

    type: Literal["max_state"] = "max_state"
    state_id: str
    value: ValueRef


class SetStateFromConditionOp(BaseModel):
    """Set a bool state variable from a condition result.

    Used for tracking previous bar state, e.g., "was price below lower band".
    """

    type: Literal["set_state_from_condition"] = "set_state_from_condition"
    state_id: str
    condition: Condition  # Forward ref, will be resolved


# Discriminated union of state operations
StateOp = Annotated[
    SetStateOp | IncrementStateOp | MaxStateOp | SetStateFromConditionOp,
    Field(discriminator="type"),
]

# Update forward refs for SetStateFromConditionOp
SetStateFromConditionOp.model_rebuild()


# =============================================================================
# Rules
# =============================================================================


class EntryRule(BaseModel):
    """Entry rule with condition, action, and optional fill hooks."""

    condition: Condition
    action: Action
    on_fill: list[StateOp] = Field(default_factory=list)


class ExitRule(BaseModel):
    """Exit rule with id, condition, action, and priority."""

    id: str
    condition: Condition
    action: Action
    priority: int = 0


class Gate(BaseModel):
    """A gate that conditionally allows/blocks other rules."""

    id: str
    condition: Condition
    target_roles: list[Literal["entry", "exit"]] = Field(default_factory=lambda: ["entry"])
    mode: Literal["allow", "block"] = "allow"


# =============================================================================
# Top-level Strategy IR
# =============================================================================


class StrategyIR(BaseModel):
    """Complete strategy intermediate representation.

    This is the output of the translator and input to the LEAN runtime.
    """

    # Metadata
    strategy_id: str
    strategy_name: str
    symbol: str
    resolution: Resolution = Resolution.HOUR

    # Indicators to create
    indicators: list[Indicator] = Field(default_factory=list)

    # State variables to track
    state: list[StateVar] = Field(default_factory=list)

    # Gates (evaluated before entry/exit)
    gates: list[Gate] = Field(default_factory=list)

    # Entry rule
    entry: EntryRule | None = None

    # Exit rules (evaluated in priority order)
    exits: list[ExitRule] = Field(default_factory=list)

    # Hooks called every bar (for state tracking, e.g., was_below_lower)
    on_bar: list[StateOp] = Field(default_factory=list)

    # Hooks called every bar when invested
    on_bar_invested: list[StateOp] = Field(default_factory=list)

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> StrategyIR:
        """Deserialize from JSON string."""
        return cls.model_validate_json(json_str)
