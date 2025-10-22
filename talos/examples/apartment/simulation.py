"""Utilities to run the apartment market simulation."""

from __future__ import annotations

from typing import Optional

from ...core.simulation import Simulation, SimulationResult
from .config import apartment_simulation_config
from .market import ApartmentMarket, FixedSupplyApartmentMarket


def run_apartment_simulation(
	*,
	buyer_count: int = 64,
	seller_count: int = 48,
	seed: Optional[int] = 1234,
	log_trades: bool = True,
) -> SimulationResult:
	market = ApartmentMarket.from_random(
		buyer_count=buyer_count,
		seller_count=seller_count,
		seed=seed,
	)
	config = apartment_simulation_config(
		log_trades=log_trades,
		random_seed=seed,
	)
	simulation = Simulation(market=market, config=config)
	return simulation.run()


def run_fixed_supply_apartment_simulation(
	*,
	buyer_count: int = 72,
	supply_capacity: float = 48.0,
	seed: Optional[int] = 1234,
	log_trades: bool = True,
) -> SimulationResult:
	market = FixedSupplyApartmentMarket.from_random(
		supply_capacity=supply_capacity,
		buyer_count=buyer_count,
		seed=seed,
	)
	config = apartment_simulation_config(
		log_trades=log_trades,
		random_seed=seed,
	)
	simulation = Simulation(market=market, config=config)
	return simulation.run()
