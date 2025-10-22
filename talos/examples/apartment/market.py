"""Apartment market implementation."""

from __future__ import annotations

from random import Random
from typing import Iterable, Optional

from ...core.market import FixedSupplyMarket, Market
from .agents import ApartmentHousehold, ApartmentLandlord, random_household, random_landlord


class ApartmentMarket(Market):
    """Market populated with apartment seekers and landlords."""

    def __init__(
        self,
        buyers: Optional[Iterable[ApartmentHousehold]] = None,
        sellers: Optional[Iterable[ApartmentLandlord]] = None,
        *,
        device: str = "cpu",
    ) -> None:
        super().__init__(buyers=buyers, sellers=sellers, device=device)

    @classmethod
    def from_random(
        cls,
        buyer_count: int = 64,
        seller_count: int = 48,
        *,
        seed: Optional[int] = None,
        device: str = "cpu",
    ) -> "ApartmentMarket":
        rng = Random(seed)
        buyers = [random_household(rng, idx, device) for idx in range(buyer_count)]
        sellers = [random_landlord(rng, idx, device) for idx in range(seller_count)]
        return cls(buyers=buyers, sellers=sellers, device=device)


class FixedSupplyApartmentMarket(FixedSupplyMarket):
    """Apartment market with exogenous supply capacity."""

    def __init__(
        self,
        supply_capacity: float,
        *,
        buyers: Optional[Iterable[ApartmentHousehold]] = None,
        device: str = "cpu",
    ) -> None:
        sellers: list[ApartmentLandlord] = []
        super().__init__(
            supply_capacity=supply_capacity,
            buyers=list(buyers) if buyers is not None else None,
            sellers=sellers,
            device=device,
        )

    @classmethod
    def from_random(
        cls,
        supply_capacity: float,
        buyer_count: int = 64,
        *,
        seed: Optional[int] = None,
        device: str = "cpu",
    ) -> "FixedSupplyApartmentMarket":
        rng = Random(seed)
        buyers = [random_household(rng, idx, device) for idx in range(buyer_count)]
        return cls(supply_capacity=supply_capacity, buyers=buyers, device=device)
