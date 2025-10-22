"""Core configuration objects for Talos simulations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass(slots=True)
class DeviceConfig:
	"""Configuration for the accelerator or host device."""

	device: str = "cpu"


@dataclass(slots=True)
class CurveSamplingConfig:
	"""Settings used when sampling demand and supply curves."""

	price_min: float = 0.25
	price_max: float = 5.0
	points: int = 64
	verify: bool = True
	verify_samples: int = 5
	verify_tolerance: float = 1e-6


@dataclass(slots=True)
class SimulationConfig:
	"""High level simulation settings shared across markets."""

	price_init: float = 1.0
	max_steps: int = 250
	tolerance: float = 1e-3
	price_tolerance: float = 1e-3
	adjustment_rate: float = 0.05
	price_bounds: Tuple[float, float] = (0.05, 25.0)
	log_trades: bool = False
	random_seed: Optional[int] = None
	device: DeviceConfig = field(default_factory=DeviceConfig)
	curve_sampling: CurveSamplingConfig = field(default_factory=CurveSamplingConfig)
	record_state: bool = True
	show_progress: bool = True
	progress_desc: str = "Tatonnement"
	curve_progress_desc: str = "Curve sampling"

	def clamp_price(self, value: float) -> float:
		lower, upper = self.price_bounds
		return max(lower, min(upper, value))


@dataclass(slots=True)
class VisualizationConfig:
	"""Settings for producing interactive market visualizations."""

	output_path: Optional[str] = None
	animation_interval_ms: int = 250
	show_equilibrium: bool = True
	show_trade_log: bool = True
