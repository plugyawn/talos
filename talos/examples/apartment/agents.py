"""Apartment market specific agents."""

from __future__ import annotations

from random import Random

from tinygrad import Tensor

from ...core.agents import BuyerAgent, LinearPreferenceModel, SellerAgent


def _household_model(device: str) -> LinearPreferenceModel:
    weights = Tensor([2.1, 1.0, 0.8], device=device)
    bias = Tensor([-0.35], device=device)
    return LinearPreferenceModel(weights=weights, bias=bias)


def _landlord_model(device: str) -> LinearPreferenceModel:
    weights = Tensor([0.9, 1.35, -0.9], device=device)
    bias = Tensor([0.55], device=device)
    return LinearPreferenceModel(weights=weights, bias=bias)


class ApartmentHousehold(BuyerAgent):
    """Buyer configured for the apartment market."""

    def __init__(
        self,
        agent_id: str,
        *,
        quantity: float = 1.0,
        income: float = 45_000.0,
        commute_penalty: float = 0.4,
        size_need: float = 1.0,
        device: str = "cpu",
    ) -> None:
        features = [income / 100_000.0, 1.0 - commute_penalty, size_need]
        model = _household_model(device)
        super().__init__(
            agent_id=agent_id,
            quantity=quantity,
            features=features,
            model=model,
            device=device,
        )
        self.income = income
        self.commute_penalty = commute_penalty
        self.size_need = size_need


class ApartmentLandlord(SellerAgent):
    """Seller configured for the apartment market."""

    def __init__(
        self,
        agent_id: str,
        *,
        quantity: float = 1.0,
        unit_quality: float = 0.6,
        maintenance_cost: float = 0.35,
        vacancy_risk: float = 0.15,
        device: str = "cpu",
    ) -> None:
        features = [unit_quality, maintenance_cost, vacancy_risk]
        model = _landlord_model(device)
        super().__init__(
            agent_id=agent_id,
            quantity=quantity,
            features=features,
            model=model,
            device=device,
        )
        self.unit_quality = unit_quality
        self.maintenance_cost = maintenance_cost
        self.vacancy_risk = vacancy_risk


def random_household(rng: Random, idx: int, device: str) -> ApartmentHousehold:
    return ApartmentHousehold(
        agent_id=f"hh-{idx}",
        income=rng.uniform(22_000.0, 110_000.0),
        commute_penalty=rng.uniform(0.05, 0.85),
        size_need=rng.uniform(0.7, 1.5),
        device=device,
    )


def random_landlord(rng: Random, idx: int, device: str) -> ApartmentLandlord:
    return ApartmentLandlord(
        agent_id=f"ll-{idx}",
        unit_quality=rng.uniform(0.45, 0.95),
        maintenance_cost=rng.uniform(0.22, 0.6),
        vacancy_risk=rng.uniform(0.04, 0.3),
        device=device,
    )
