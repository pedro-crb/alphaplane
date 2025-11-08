from __future__ import annotations

from alphaplane.airfoil.foil_condition import AirfoilCondition
from alphaplane.analysis_tools.analysis_data import AnalysisData


class PolarData(AnalysisData):
    def __init__(self):
        super().__init__(
            AirfoilCondition,
            ['Cl', 'Cd', 'Cdp', 'Cm', 'xtr_top_result', 'xtr_bottom_result', 'Cp_min', 'M_crit', 'confidence'],
            True
        )

    @property
    def condition_structured(self) -> AirfoilCondition | None:
        return_value = super().condition_structured
        assert isinstance(return_value, AirfoilCondition) or return_value is None
        return return_value

    @property
    def condition_unstructured(self) -> AirfoilCondition | None:
        return_value = super().condition_unstructured
        assert isinstance(return_value, AirfoilCondition) or return_value is None
        return return_value

    @property
    def condition(self) -> AirfoilCondition | None:
        return_value = super().condition
        assert isinstance(return_value, AirfoilCondition) or return_value is None
        return return_value


class BoundaryLayerData(AnalysisData):
    def __init__(self):
        super().__init__(
            AirfoilCondition,
            ['s', 'x', 'y', 'vel_ratio', 'dstar', 'theta', 'Cf', 'H', 'Cp'],
            False
        )

    @property
    def condition_structured(self):
        return super().condition_structured

    @property
    def condition_unstructured(self):
        return super().condition_unstructured

    @property
    def condition(self):
        return super().condition
