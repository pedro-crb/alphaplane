from __future__ import annotations
from typing import Self


class Battery:
    def __init__(self, volts: float,
                 resistance: float,
                 charge_mAh: float | None = None,
                 max_amps: float | None = None,
                 weight: float | None = None,
                 n_cells: int | None = None,
                 ) -> None:
        self.volts: float = volts
        self.resistance: float = resistance
        self.charge_mAh: float | None = charge_mAh
        self.max_amps: float | None = max_amps
        self.weight: float | None = weight
        self.n_cells: int | None = n_cells

    @property
    def volts_per_cell(self) -> float | None:
        return self.volts/self.n_cells if self.n_cells is not None else None

    @property
    def resistance_per_cell(self) -> float | None:
        return self.resistance / self.n_cells if self.n_cells is not None else None

    @property
    def max_watts(self) -> float | None:
        return self.max_amps*self.volts if self.max_amps is not None else None

    @classmethod
    def from_ideal_battery(cls, volts) -> Self:
        return cls(volts, 0.0)

    @classmethod
    def from_lipo(cls, num_cells: int, charge_mAh: float | None = None, max_amps: float | None = None,
                  volts_per_cell: float = 4.0, resistance_per_cell: float = 0.005):
        volts = volts_per_cell * num_cells
        resistance = resistance_per_cell * num_cells
        weight = 0.02 + 24*1e-6*charge_mAh*num_cells if charge_mAh is not None else None
        return cls(volts, resistance, charge_mAh, max_amps, weight, num_cells)
