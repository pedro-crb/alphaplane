from __future__ import annotations

from alphaplane.propeller.prop_condition import PropellerCondition
from alphaplane.analysis_tools.analysis_data import AnalysisData


class PropellerData(AnalysisData):
    def __init__(self):
        super().__init__(PropellerCondition, ['CT', 'CP'])

    @property
    def condition_structured(self) -> PropellerCondition | None:
        return_value = super().condition_structured
        assert isinstance(return_value, PropellerCondition) or return_value is None
        return return_value

    @property
    def condition_unstructured(self) -> PropellerCondition | None:
        return_value = super().condition_unstructured
        assert isinstance(return_value, PropellerCondition) or return_value is None
        return return_value
    @property
    def condition(self) -> PropellerCondition | None:
        return_value = super().condition
        assert isinstance(return_value, PropellerCondition) or return_value is None
        return return_value