from dataclasses import dataclass


@dataclass
class FaultOrder:
    name: str = "Order"
    fundamental: float = 1.0
    n_harmonics: int = 5
    color: str = "#e6194b"
