"""Simulation loop orchestrating market interactions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional

from tinygrad import Tensor
from tqdm.auto import tqdm

from .config import SimulationConfig
from .market import CurveData, Market, MarketObservation, TradeRecord


@dataclass(slots=True)
class SimulationHistory:
    observations: List[MarketObservation] = field(default_factory=list)
    trades: List[TradeRecord] = field(default_factory=list)
    curves: Optional[CurveData] = None

    def last_observation(self) -> Optional[MarketObservation]:
        return self.observations[-1] if self.observations else None


@dataclass(slots=True)
class SimulationResult:
    market: Market
    config: SimulationConfig
    history: SimulationHistory

    @property
    def equilibrium(self) -> Optional[MarketObservation]:
        return self.history.last_observation()


class Simulation:
    """Runs tatonnement style price adjustment for a market."""

    def __init__(self, market: Market, config: Optional[SimulationConfig] = None) -> None:
        self.market = market
        self.config = config or SimulationConfig()

    def _price_tensor(self, value: float) -> Tensor:
        return Tensor([value], device=self.market.device)

    def _linspace(self, start: float, end: float, count: int) -> List[float]:
        if count <= 1:
            return [start]
        step = (end - start) / (count - 1)
        return [start + step * idx for idx in range(count)]

    def run(self) -> SimulationResult:
        history = SimulationHistory()
        config = self.config
        price_value = config.clamp_price(config.price_init)
        price_tensor = self._price_tensor(price_value)

        curve_prices = self._linspace(
            config.curve_sampling.price_min,
            config.curve_sampling.price_max,
            config.curve_sampling.points,
        )
        history.curves = (
            self.market.sample_curves(
                curve_prices,
                show_progress=config.show_progress,
                description=config.curve_progress_desc,
                verify=config.curve_sampling.verify,
                verify_samples=config.curve_sampling.verify_samples,
                verify_tolerance=config.curve_sampling.verify_tolerance,
            )
            if config.record_state
            else None
        )

        lower_price, upper_price = config.price_bounds
        lower_price = config.clamp_price(lower_price)
        upper_price = config.clamp_price(upper_price)
        if upper_price <= lower_price:
            raise ValueError("Simulation price bounds must satisfy lower < upper.")

        def evaluate_excess_at(price_val: float) -> float:
            probe_tensor = self._price_tensor(price_val)
            state = self.market.build_state(-1, probe_tensor)
            demand_val = self.market.aggregate_demand(probe_tensor, state)
            supply_val = self.market.aggregate_supply(probe_tensor, state)
            return demand_val - supply_val

        lower_excess = evaluate_excess_at(lower_price)
        upper_excess = evaluate_excess_at(upper_price)
        bracket_warning_issued = False

        converged = False
        last_excess: Optional[float] = None
        step_progress = None
        if config.show_progress:
            step_progress = tqdm(
                range(config.max_steps),
                desc=config.progress_desc,
                total=config.max_steps,
            )
            step_iterable: Iterable[int] = step_progress
        else:
            step_iterable = range(config.max_steps)

        try:
            for step in step_iterable:
                state = self.market.build_state(step, price_tensor)
                demand_value = self.market.aggregate_demand(price_tensor, state)
                supply_value = self.market.aggregate_supply(price_tensor, state)
                excess_value = demand_value - supply_value
                last_excess = excess_value

                observation = MarketObservation(
                    step=step,
                    price=price_value,
                    demand=demand_value,
                    supply=supply_value,
                    excess=excess_value,
                )
                history.observations.append(observation)

                if config.log_trades:
                    trade = self.market.execute_trades(step, price_tensor, state, log_details=True)
                    history.trades.append(trade)

                if step_progress is not None:
                    step_progress.set_postfix(
                        price=f"{price_value:.3f}",
                        excess=f"{excess_value:.3f}",
                    )

                if abs(excess_value) <= config.tolerance:
                    converged = True
                    if config.show_progress:
                        tqdm.write(
                            f"Equilibrium reached at step {step} "
                            f"price {price_value:.3f} excess {excess_value:.3f}"
                        )
                    break

                if excess_value > 0:
                    lower_price = max(lower_price, price_value)
                    lower_excess = excess_value
                elif excess_value < 0:
                    upper_price = min(upper_price, price_value)
                    upper_excess = excess_value

                bracket_ready = (
                    lower_excess is not None
                    and upper_excess is not None
                    and lower_excess > 0
                    and upper_excess < 0
                )

                if not bracket_ready and not bracket_warning_issued:
                    if config.show_progress:
                        tqdm.write(
                            "Tatonnement running without a valid price bracket; "
                            "fallback adjustments may oscillate."
                        )
                    bracket_warning_issued = True

                if bracket_ready and (upper_price - lower_price) <= config.price_tolerance:
                    converged = True
                    final_price = config.clamp_price((lower_price + upper_price) * 0.5)
                    if config.show_progress:
                        tqdm.write(
                            f"Equilibrium bracketed within {upper_price - lower_price:.4f} price units; "
                            f"estimated price {final_price:.3f}."
                        )
                    if abs(final_price - price_value) > config.price_tolerance:
                        final_tensor = self._price_tensor(final_price)
                        final_state = self.market.build_state(step, final_tensor)
                        final_demand = self.market.aggregate_demand(final_tensor, final_state)
                        final_supply = self.market.aggregate_supply(final_tensor, final_state)
                        final_excess = final_demand - final_supply
                        history.observations.append(
                            MarketObservation(
                                step=step + 1,
                                price=final_price,
                                demand=final_demand,
                                supply=final_supply,
                                excess=final_excess,
                            )
                        )
                    price_value = final_price
                    price_tensor = self._price_tensor(price_value)
                    break

                if bracket_ready:
                    next_price = (lower_price + upper_price) * 0.5
                else:
                    next_price = config.clamp_price(
                        price_value + excess_value * config.adjustment_rate
                    )

                if abs(next_price - price_value) <= config.price_tolerance:
                    converged = True
                    price_value = next_price
                    price_tensor = self._price_tensor(price_value)
                    break

                price_value = next_price
                price_tensor = self._price_tensor(price_value)
        finally:
            if step_progress is not None:
                step_progress.close()

        if not converged and config.show_progress:
            trailing_excess = last_excess if last_excess is not None else float("nan")
            tqdm.write(
                f"Reached max steps ({config.max_steps}) without convergence; "
                f"last excess={trailing_excess:.3f}."
            )

        if config.log_trades and len(history.trades) < len(history.observations):
            last_step = history.observations[-1].step
            last_price = self._price_tensor(history.observations[-1].price)
            state = self.market.build_state(last_step, last_price)
            history.trades.append(
                self.market.execute_trades(last_step, last_price, state, log_details=True)
            )

        return SimulationResult(market=self.market, config=config, history=history)
