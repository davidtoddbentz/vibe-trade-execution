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
- StrategyIR

Defined locally (LEAN-specific):
- Resolution enum (LEAN timeframe)
- Indicator types (EMA, SMA, BollingerBands, etc.) - discriminated union for LEAN runtime
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal

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
    LiteralRef,
    MarketOrderAction,
    MaxStateAction,
    MinStateAction,
    NotCondition,
    OverlayRule,
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
# LEAN-specific: Resolution Enum
# =============================================================================


class Resolution(str, Enum):
    """Trading resolution/timeframe for LEAN."""

    MINUTE = "Minute"
    HOUR = "Hour"
    DAILY = "Daily"


# =============================================================================
# LEAN-specific: Indicator Types
# These are discriminated union types for the LEAN runtime, which expects
# specific indicator configurations rather than generic IndicatorSpec.
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


class RSI(BaseModel):
    """Relative Strength Index."""

    type: Literal["RSI"] = "RSI"
    id: str
    period: int = 14


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
    symbol: str | None = None  # If set, use this symbol's data (for intermarket)


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


class VWAPBands(BaseModel):
    """VWAP with standard deviation bands.

    Provides upper/middle/lower bands similar to Bollinger Bands,
    but centered on VWAP instead of SMA.
    """

    type: Literal["VWAP_BANDS"] = "VWAP_BANDS"
    id: str
    anchor: Literal["session", "week", "month", "ytd", "custom"] = "session"
    multiplier: float = 2.0
    anchor_datetime: str | None = None  # ISO format datetime for custom anchor


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


class SessionHighLow(BaseModel):
    """Session high/low tracker.

    Tracks the highest high or lowest low within a trading session.
    Resets at session open.
    """

    type: Literal["SESSION_HL"] = "SESSION_HL"
    id: str
    mode: Literal["high", "low"]
    session: str = "us_equity"  # Session identifier for reset timing


class AnchoredVWAP(BaseModel):
    """Anchored VWAP indicator.

    Calculates VWAP anchored to a specific point in time (session open, week start,
    custom datetime, etc.) along with standard deviation bands.

    Fields:
        - value: The VWAP value
        - std_dev: Standard deviation from VWAP
    """

    type: Literal["AVWAP"] = "AVWAP"
    id: str
    anchor: Literal["session", "session_open", "week", "week_open", "month", "month_start", "ytd", "custom"] = "session"
    anchor_datetime: str | None = None  # ISO format datetime for custom anchor


class Percentile(BaseModel):
    """Rolling percentile rank indicator.

    Calculates where the current value of a metric falls within its historical distribution.
    Used for squeeze detection (BB width percentile), volume percentile, etc.

    Fields:
        - value: The percentile rank (0-100)
    """

    type: Literal["PCTILE"] = "PCTILE"
    id: str
    period: int = 100  # Lookback period for percentile calculation
    percentile: float = 10.0  # Target percentile level
    source: str = "close"  # Source field for percentile calculation (close, bb_width, volume, etc.)


class Gap(BaseModel):
    """Gap detection indicator.

    Tracks the gap between previous session close and current session open.
    Used for gap trading strategies (gap_fade, gap_go).

    Fields:
        - gap_pct: Gap percentage (open - prev_close) / prev_close * 100
        - direction: Gap direction ('up' or 'down')
    """

    type: Literal["GAP"] = "GAP"
    id: str
    session: str = "us"  # Session for gap calculation


# Discriminated union of all indicator types
Indicator = Annotated[
    EMA
    | SMA
    | BollingerBands
    | KeltnerChannel
    | ATR
    | RSI
    | Maximum
    | Minimum
    | RateOfChange
    | ADX
    | DonchianChannel
    | VWAP
    | VWAPBands
    | RollingWindow
    | VolumeSMA
    | RollingMinMax
    | SessionHighLow
    | AnchoredVWAP
    | Percentile
    | Gap,
    Field(discriminator="type"),
]


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
