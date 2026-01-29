"""Intermediate Representation (IR) for strategy translation.

This module re-exports IR types from the shared library (single source of truth)
and defines only execution-specific types locally.

From shared library:
- All enums (CompareOp, PriceField, BandField, StateType, IndicatorProperty)
- All value references (IndicatorRef, PriceRef, StateRef, etc.)
- All conditions (CompareCondition, AllOfCondition, RegimeCondition, etc.)
- All actions (SetHoldingsAction, LiquidateAction, MarketOrderAction)
- All state operations (SetStateAction, IncrementStateAction, etc.)
- All rules (EntryRule, ExitRule, GateRule, OverlayRule)
- All indicator types (EMA, SMA, BollingerBands, etc.)
- StrategyIR

Defined locally (LEAN-specific):
- Resolution enum (LEAN timeframe)
- StrategyIR (extends shared with LEAN-specific typed indicators and Resolution)
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

# =============================================================================
# Re-export everything from shared library
# =============================================================================
from vibe_trade_shared.models.ir import (  # noqa: F401
    # Actions
    Action,
    # Conditions
    AllOfCondition,
    AnyOfCondition,
    # Enums
    BandField,
    BreakoutCondition,
    CompareCondition,
    CompareOp,
    Condition,
    CrossCondition,
    CrossDirection,
    EntryAction,
    # Rules
    EntryRule,
    EventWindowCondition,
    ExitAction,
    ExitRule,
    GateRule,
    # State operations
    IncrementStateAction,
    # Value references
    IndicatorBandRef,
    IndicatorProperty,
    IndicatorPropertyRef,
    IndicatorRef,
    # Specs
    IndicatorSpec,
    IntermarketCondition,
    IRExpression,
    LiquidateAction,
    ReducePositionAction,
    LiteralRef,
    MarketOrderAction,
    MaxStateAction,
    MinStateAction,
    NotCondition,
    OverlayRule,
    PositionPolicy,
    PriceField,
    PriceRef,
    RegimeCondition,
    RollingWindowRef,
    SequenceCondition,
    SequenceStep,
    SetHoldingsAction,
    SetStateAction,
    SetStateFromConditionAction,
    SpreadCondition,
    SqueezeCondition,
    StateCondition,
    StateOp,
    StateRef,
    StateType,
    StateVarSpec,
    TimeFilterCondition,
    TimeRef,
    ValueRef,
    VolumeRef,
)

# =============================================================================
# Re-export indicator types from shared library
# =============================================================================
from vibe_trade_shared.models.ir.indicators import (  # noqa: F401
    ADX,
    ATR,
    EMA,
    RSI,
    SMA,
    VWAP,
    AnchoredVWAP,
    BollingerBands,
    DonchianChannel,
    Gap,
    Indicator,
    KeltnerChannel,
    Maximum,
    Minimum,
    Percentile,
    RateOfChange,
    RollingMinMax,
    RollingWindow,
    SessionHighLow,
    VolumeSMA,
    VWAPBands,
    IndicatorUnion,
    indicator_id_from_spec,
)

# =============================================================================
# LEAN-specific: Resolution Enum
# =============================================================================


class Resolution(str, Enum):
    """Trading resolution/timeframe for LEAN."""

    MINUTE = "Minute"
    HOUR = "Hour"
    DAILY = "Daily"


# =============================================================================
# LEAN-specific: StrategyIR with typed indicators
# The shared library uses IndicatorSpec (generic), but LEAN needs specific types.
# =============================================================================


class StrategyIR(BaseModel):
    """Complete strategy intermediate representation for LEAN execution.

    This extends the shared library's StrategyIR with LEAN-specific indicator types.
    """

    # Metadata
    strategy_id: str
    strategy_name: str
    symbol: str
    resolution: Resolution = Resolution.HOUR

    # Additional symbols to subscribe to (for intermarket strategies)
    additional_symbols: list[str] = Field(default_factory=list)

    # Indicators to create (LEAN-specific typed union)
    indicators: list[Indicator] = Field(default_factory=list)

    # State variables to track
    state: list[StateVarSpec] = Field(default_factory=list)

    # Gates (evaluated before entry/exit)
    gates: list[GateRule] = Field(default_factory=list)

    # Overlays (scale risk/size based on conditions)
    overlays: list[OverlayRule] = Field(default_factory=list)

    # Entry rule
    entry: EntryRule | None = None

    # Exit rules (evaluated in priority order)
    exits: list[ExitRule] = Field(default_factory=list)

    # Hooks called every bar (for state tracking)
    on_bar: list[StateOp] = Field(default_factory=list)

    # Hooks called every bar when invested
    on_bar_invested: list[StateOp] = Field(default_factory=list)

    # Trading costs (for backtest simulation)
    fee_pct: float = Field(
        default=0.0,
        ge=0.0,
        le=5.0,
        description="Trading fee as percentage of trade value (e.g., 0.1 = 0.1%)",
    )
    slippage_pct: float = Field(
        default=0.0,
        ge=0.0,
        le=5.0,
        description="Slippage as percentage of price (e.g., 0.05 = 0.05%)",
    )

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> StrategyIR:
        """Deserialize from JSON string."""
        return cls.model_validate_json(json_str)
