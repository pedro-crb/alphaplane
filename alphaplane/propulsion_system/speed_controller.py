from __future__ import annotations


class SpeedController:
    def __init__(self,
                 resistance: float | None = None,
                 max_amps: float | None = None,
                 max_volts: float | None = None,
                 max_watts: float | None = None,
                 weight: float | None = None,
                 ) -> None:
        self.resistance: float = resistance if resistance is not None else 0.0
        self.max_amps = max_amps
        self.max_volts = max_volts
        self.max_watts = max_watts
        self.weight = weight
