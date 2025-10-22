"""Visualization helper for the apartment market example."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

from ...core.config import VisualizationConfig
from ...core.visualization import MarketVisualizer
from .simulation import run_apartment_simulation, run_fixed_supply_apartment_simulation


def render_apartment_animation(
	output_path: Optional[str | Path] = None,
	*,
	buyer_count: int = 64,
	seller_count: int = 48,
	seed: Optional[int] = 1234,
	output_dir: str | Path = Path("artifacts"),
	filename_prefix: str = "apartment_market_animation",
	timestamp: Optional[str] = None,
) -> Path:
	"""Run the apartment simulation and persist the Plotly animation."""

	result = run_apartment_simulation(
		buyer_count=buyer_count,
		seller_count=seller_count,
		seed=seed,
	)

	if output_path is None:
		stamp = timestamp or datetime.now().strftime("%Y%m%d-%H%M%S")
		target_dir = Path(output_dir)
		target_dir.mkdir(parents=True, exist_ok=True)
		final_path = target_dir / f"{filename_prefix}_{stamp}.html"
	else:
		final_path = Path(output_path)
		final_path.parent.mkdir(parents=True, exist_ok=True)

	viz_config = VisualizationConfig(output_path=str(final_path), show_trade_log=True)
	visualizer = MarketVisualizer(config=viz_config)
	visualizer.build_animation(result)
	return final_path


def render_fixed_supply_animation(
	output_path: Optional[str | Path] = None,
	*,
	buyer_count: int = 72,
	supply_capacity: float = 48.0,
	seed: Optional[int] = 1234,
	output_dir: str | Path = Path("artifacts"),
	filename_prefix: str = "apartment_fixed_supply_animation",
	timestamp: Optional[str] = None,
) -> Path:
	"""Render an animation for the fixed-supply apartment scenario."""

	result = run_fixed_supply_apartment_simulation(
		buyer_count=buyer_count,
		supply_capacity=supply_capacity,
		seed=seed,
	)

	if output_path is None:
		stamp = timestamp or datetime.now().strftime("%Y%m%d-%H%M%S")
		target_dir = Path(output_dir)
		target_dir.mkdir(parents=True, exist_ok=True)
		final_path = target_dir / f"{filename_prefix}_{stamp}.html"
	else:
		final_path = Path(output_path)
		final_path.parent.mkdir(parents=True, exist_ok=True)

	viz_config = VisualizationConfig(output_path=str(final_path), show_trade_log=True)
	visualizer = MarketVisualizer(config=viz_config)
	visualizer.build_animation(result)
	return final_path
