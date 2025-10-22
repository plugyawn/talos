"""Agent abstractions built on top of tinygrad tensors."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Optional, Protocol, Sequence

from tinygrad import Tensor


TensorLike = float | Sequence[float] | Tensor


def _as_tensor(value: TensorLike, device: str) -> Tensor:
    if isinstance(value, Tensor):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return Tensor(list(value), device=device)
    return Tensor([float(value)], device=device)


class PreferenceModel(Protocol):
    """Callable contract used to score an agent's reservation price."""

    def __call__(self, features: Optional[Tensor], market_state: Dict[str, Tensor]) -> Tensor:
        ...


@dataclass(slots=True)
class LinearPreferenceModel:
    """Simple linear preference model using tinygrad tensors."""

    weights: Optional[Tensor] = None
    bias: Tensor = field(default_factory=lambda: Tensor([0.0]))

    def __call__(self, features: Optional[Tensor], market_state: Dict[str, Tensor]) -> Tensor:
        if features is None:
            return self.bias
        if self.weights is None:
            raise ValueError("Weights must be provided when features are present.")
        return features @ self.weights + self.bias


@dataclass(slots=True)
class EconomicAgent:
    """Base class for agents participating in a market."""

    agent_id: str
    quantity: float = 1.0
    features: Optional[TensorLike] = None
    model: Optional[PreferenceModel] = None
    device: str = "cpu"

    def __post_init__(self) -> None:
        self._quantity_tensor = _as_tensor(self.quantity, self.device)
        self._features_tensor = _as_tensor(self.features, self.device) if self.features is not None else None
        if self.model is None:
            self.model = LinearPreferenceModel(bias=Tensor([1.0], device=self.device))

    def reservation_price(self, market_state: Dict[str, Tensor]) -> Tensor:
        if self.model is None:
            raise ValueError("Preference model is not defined for this agent.")
        return self.model(self._features_tensor, market_state)

    def tensor_quantity(self) -> Tensor:
        return self._quantity_tensor

    def update_context(self, **kwargs: Any) -> None:
        features = kwargs.get("features")
        if features is not None:
            self._features_tensor = _as_tensor(features, self.device)


@dataclass(slots=True)
class BuyerAgent(EconomicAgent):
    """Represents an agent looking to buy a unit."""

    def demand_at_price(self, price: Tensor, market_state: Dict[str, Tensor]) -> Tensor:
        reservation = self.reservation_price(market_state)
        return Tensor.where(price <= reservation, self.tensor_quantity(), Tensor([0.0], device=self.device))

    def willing_to_trade(self, price: Tensor, market_state: Dict[str, Tensor]) -> bool:
        demand = self.demand_at_price(price, market_state)
        return bool(demand.item() > 0)


@dataclass(slots=True)
class SellerAgent(EconomicAgent):
    """Represents an agent offering a unit for sale."""

    def supply_at_price(self, price: Tensor, market_state: Dict[str, Tensor]) -> Tensor:
        reservation = self.reservation_price(market_state)
        return Tensor.where(price >= reservation, self.tensor_quantity(), Tensor([0.0], device=self.device))

    def willing_to_trade(self, price: Tensor, market_state: Dict[str, Tensor]) -> bool:
        supply = self.supply_at_price(price, market_state)
        return bool(supply.item() > 0)
