"""Market abstractions built on top of tinygrad tensors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from tinygrad import Tensor
from tqdm.auto import tqdm

from .agents import BuyerAgent, SellerAgent


@dataclass(slots=True)
class TradeRecord:
	"""Represents an executed trade in the market."""

	step: int
	price: float
	quantity: float
	buyer_ids: List[str]
	seller_ids: List[str]


@dataclass(slots=True)
class MarketObservation:
	"""Aggregated observation for a single simulation step."""

	step: int
	price: float
	demand: float
	supply: float
	excess: float


@dataclass(slots=True)
class CurveData:
	"""Demand and supply curve samples."""

	prices: List[float]
	demand: List[float]
	supply: List[float]


class Market:
	"""Base market implementation with buyer and seller populations."""

	def __init__(
		self,
		buyers: Optional[Iterable[BuyerAgent]] = None,
		sellers: Optional[Iterable[SellerAgent]] = None,
		device: str = "cpu",
	) -> None:
		self.buyers: List[BuyerAgent] = list(buyers) if buyers is not None else []
		self.sellers: List[SellerAgent] = list(sellers) if sellers is not None else []
		self.device = device

	def register_buyer(self, agent: BuyerAgent) -> None:
		self.buyers.append(agent)

	def register_seller(self, agent: SellerAgent) -> None:
		self.sellers.append(agent)

	def build_state(self, step: int, price: Tensor) -> Dict[str, Tensor]:
		return {
			"step": Tensor([float(step)], device=self.device),
			"price": price,
		}

	def aggregate_demand(self, price: Tensor, market_state: Dict[str, Tensor]) -> float:
		total = 0.0
		for buyer in self.buyers:
			demand_tensor = buyer.demand_at_price(price, market_state)
			total += float(demand_tensor.item())
		return total

	def aggregate_supply(self, price: Tensor, market_state: Dict[str, Tensor]) -> float:
		total = 0.0
		for seller in self.sellers:
			supply_tensor = seller.supply_at_price(price, market_state)
			total += float(supply_tensor.item())
		return total

	def sample_curves(
		self,
		prices: Iterable[float],
		*,
		show_progress: bool = False,
		description: str = "Curve sampling",
		verify: bool = True,
		verify_samples: int = 5,
		verify_tolerance: float = 1e-6,
	) -> CurveData:
		price_list = list(prices)
		if not price_list:
			return CurveData(prices=[], demand=[], supply=[])

		price_tensor = Tensor(price_list, device=self.device)
		price_tensor = price_tensor.reshape(-1)
		state_vector = self.build_state(step=0, price=price_tensor)

		demand_vector = self._vectorized_demand(price_tensor, state_vector)
		supply_vector = self._vectorized_supply(price_tensor, state_vector)

		demand = [float(x) for x in demand_vector.numpy().tolist()]
		supply = [float(x) for x in supply_vector.numpy().tolist()]

		progress_bar = None
		if show_progress:
			progress_bar = tqdm(total=len(price_list), desc=description)
			progress_bar.update(len(price_list))
			excess_preview = demand[-1] - supply[-1]
			progress_bar.set_postfix(
				price=f"{price_list[-1]:.2f}",
				excess=f"{excess_preview:.3f}",
			)
			progress_bar.close()

		if verify:
			check_count = min(verify_samples, len(price_list))
			for idx in range(check_count):
				price_value = price_list[idx]
				scalar_price = Tensor([price_value], device=self.device)
				state_scalar = self.build_state(step=idx, price=scalar_price)
				demand_scalar = self.aggregate_demand(scalar_price, state_scalar)
				supply_scalar = self.aggregate_supply(scalar_price, state_scalar)
				demand_diff = abs(demand_scalar - demand[idx])
				supply_diff = abs(supply_scalar - supply[idx])
				assert (
					demand_diff <= verify_tolerance
				), f"Vectorized demand drift {demand_diff} exceeds tolerance at index {idx}."
				assert (
					supply_diff <= verify_tolerance
				), f"Vectorized supply drift {supply_diff} exceeds tolerance at index {idx}."

		return CurveData(prices=price_list, demand=demand, supply=supply)

	def _vectorized_demand(self, price_tensor: Tensor, state_vector: Dict[str, Tensor]) -> Tensor:
		total = Tensor.zeros(price_tensor.shape, device=self.device)
		for buyer in self.buyers:
			contrib = buyer.demand_at_price(price_tensor, state_vector)
			assert contrib.shape == price_tensor.shape, "Demand contribution shape mismatch."
			total = total + contrib
		return total

	def _vectorized_supply(self, price_tensor: Tensor, state_vector: Dict[str, Tensor]) -> Tensor:
		total = Tensor.zeros(price_tensor.shape, device=self.device)
		for seller in self.sellers:
			contrib = seller.supply_at_price(price_tensor, state_vector)
			assert contrib.shape == price_tensor.shape, "Supply contribution shape mismatch."
			total = total + contrib
		return total

	def execute_trades(
		self,
		step: int,
		price: Tensor,
		market_state: Dict[str, Tensor],
		log_details: bool = False,
	) -> TradeRecord:
		buyer_ids: List[str] = []
		seller_ids: List[str] = []
		buyer_quantity = 0.0
		seller_quantity = 0.0

		for buyer in self.buyers:
			if buyer.willing_to_trade(price, market_state):
				buyer_quantity += buyer.quantity
				if log_details:
					buyer_ids.append(buyer.agent_id)

		for seller in self.sellers:
			if seller.willing_to_trade(price, market_state):
				seller_quantity += seller.quantity
				if log_details:
					seller_ids.append(seller.agent_id)

		settled_quantity = min(buyer_quantity, seller_quantity)

		return TradeRecord(
			step=step,
			price=float(price.item()),
			quantity=settled_quantity,
			buyer_ids=buyer_ids,
			seller_ids=seller_ids,
		)

	def observation(self, step: int, price: Tensor) -> MarketObservation:
		state = self.build_state(step, price)
		demand = self.aggregate_demand(price, state)
		supply = self.aggregate_supply(price, state)
		return MarketObservation(
			step=step,
			price=float(price.item()),
			demand=demand,
			supply=supply,
			excess=demand - supply,
		)


class FixedSupplyMarket(Market):
	"""Market variant with exogenously fixed aggregate supply."""

	def __init__(
		self,
		supply_capacity: float,
		*,
		buyers: Optional[Iterable[BuyerAgent]] = None,
		sellers: Optional[Iterable[SellerAgent]] = None,
		device: str = "cpu",
	) -> None:
		super().__init__(buyers=buyers, sellers=sellers, device=device)
		self.supply_capacity = supply_capacity

	def aggregate_supply(self, price: Tensor, market_state: Dict[str, Tensor]) -> float:
		return self.supply_capacity

	def _vectorized_supply(self, price_tensor: Tensor, state_vector: Dict[str, Tensor]) -> Tensor:
		return Tensor.full(price_tensor.shape, self.supply_capacity, device=self.device)

	def execute_trades(
		self,
		step: int,
		price: Tensor,
		market_state: Dict[str, Tensor],
		log_details: bool = False,
	) -> TradeRecord:
		trade = super().execute_trades(step, price, market_state, log_details=log_details)
		trade.quantity = min(trade.quantity, self.supply_capacity)
		return trade
