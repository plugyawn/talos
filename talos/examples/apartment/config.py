"""Convenience factory for apartment simulation configuration."""

from __future__ import annotations

from typing import Any

from ...core.config import CurveSamplingConfig, SimulationConfig


def apartment_simulation_config(**overrides: Any) -> SimulationConfig:
	config = SimulationConfig(
		price_init=1.8,
		max_steps=220,
		tolerance=5e-3,
		adjustment_rate=0.08,
		price_bounds=(0.35, 5.0),
		log_trades=bool(overrides.pop("log_trades", False)),
	)
	config.curve_sampling = CurveSamplingConfig(price_min=0.35, price_max=5.0, points=120)

	for key, value in overrides.items():
		if hasattr(config, key):
			setattr(config, key, value)
		else:
			raise AttributeError(f"Unknown SimulationConfig attribute '{key}'")
	return config
