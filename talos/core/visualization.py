"""Plotly based animations for market simulations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import plotly.graph_objects as go

from .config import VisualizationConfig
from .simulation import SimulationResult


@dataclass(slots=True)
class MarketVisualizer:
	config: VisualizationConfig = VisualizationConfig()

	def build_animation(self, result: SimulationResult) -> go.Figure:
		history = result.history
		observations = history.observations
		if not observations:
			raise ValueError("Simulation history is empty; run the simulation first.")

		curves = history.curves
		demand_curve = go.Scatter(
			x=curves.prices if curves else [],
			y=curves.demand if curves else [],
			mode="lines",
			name="Demand",
			line=dict(color="#1f77b4"),
		)
		supply_curve = go.Scatter(
			x=curves.prices if curves else [],
			y=curves.supply if curves else [],
			mode="lines",
			name="Supply",
			line=dict(color="#ff7f0e"),
		)

		demand_points = go.Scatter(
			x=[observations[0].price],
			y=[observations[0].demand],
			mode="markers",
			marker=dict(size=12, color="#1f77b4"),
			name="Current demand",
		)

		supply_points = go.Scatter(
			x=[observations[0].price],
			y=[observations[0].supply],
			mode="markers",
			marker=dict(size=12, color="#ff7f0e"),
			name="Current supply",
		)

		frames: List[go.Frame] = []
		for obs in observations:
			frames.append(
				go.Frame(
					data=[
						go.Scatter(x=curves.prices if curves else [], y=curves.demand if curves else []),
						go.Scatter(x=curves.prices if curves else [], y=curves.supply if curves else []),
						go.Scatter(x=[obs.price], y=[obs.demand]),
						go.Scatter(x=[obs.price], y=[obs.supply]),
					],
					name=f"step-{obs.step}",
					layout=go.Layout(title=f"Price {obs.price:.2f} | Step {obs.step}"),
				)
			)

		fig = go.Figure(data=[demand_curve, supply_curve, demand_points, supply_points], frames=frames)
		fig.update_layout(
			title="Market Adjustment",
			xaxis_title="Price",
			yaxis_title="Quantity",
			updatemenus=[
				{
					"type": "buttons",
					"buttons": [
						{"label": "Play", "method": "animate", "args": [None, {"frame": {"duration": self.config.animation_interval_ms, "redraw": True}, "fromcurrent": True}]},
						{"label": "Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0}, "mode": "immediate", "transition": {"duration": 0}}]},
					],
				}
			],
			sliders=[
				{
					"steps": [
						{
							"label": str(obs.step),
							"method": "animate",
							"args": [[f"step-{obs.step}"], {"frame": {"duration": 0}, "mode": "immediate", "transition": {"duration": 0}}],
						}
						for obs in observations
					]
				}
			],
		)

		if self.config.show_equilibrium and observations:
			fig.add_vline(x=observations[-1].price, line_dash="dash", line_color="#2ca02c")

		if self.config.show_trade_log and result.history.trades:
			log_lines = [
				f"Step {trade.step}: price {trade.price:.2f}, qty {trade.quantity:.2f}"
				for trade in result.history.trades
			]
			fig.update_layout(
				annotations=[
					dict(
						text="<br>".join(log_lines),
						xref="paper",
						yref="paper",
						x=1.02,
						y=1.0,
						showarrow=False,
						align="left",
						bordercolor="#bbbbbb",
						borderwidth=1,
						bgcolor="#f9f9f9",
					)
				]
			)

		output_path = self.config.output_path
		if output_path:
			fig.write_html(output_path, include_plotlyjs="cdn")

		return fig
