from __future__ import annotations

import numpy as np

from alphaplane.analysis_tools.analysis_condition import AnalysisCondition


class AirfoilCondition(AnalysisCondition):
    def __init__(self, alpha: float | np.ndarray | list[float] | None = None,
                 reynolds: float | np.ndarray | list[float] | None = None,
                 mach: float | np.ndarray | list[float] | None = None,
                 deflection: float | np.ndarray | list[float] | None = None,
                 hinge_position: float | np.ndarray | list[float] | None = None,
                 xtr_top: float | np.ndarray | list[float] | None = None,
                 xtr_bottom: float | np.ndarray | list[float] | None = None,
                 n_crit: float | np.ndarray | list[float] | None = None,
                 is_sequence: bool = False,
                 ) -> None:
        super().__init__()
        self.alpha: np.ndarray | None = np.atleast_1d(alpha) if alpha is not None else None
        self.reynolds: np.ndarray | None = np.atleast_1d(reynolds) if reynolds is not None else None
        self.mach: np.ndarray | None = np.atleast_1d(mach) if mach is not None else None
        self.deflection: np.ndarray | None = np.atleast_1d(deflection) if deflection is not None else None
        self.hinge_position: np.ndarray | None = np.atleast_1d(hinge_position) if hinge_position is not None else None
        self.xtr_top: np.ndarray | None = np.atleast_1d(xtr_top) if xtr_top is not None else None
        self.xtr_bottom: np.ndarray | None = np.atleast_1d(xtr_bottom) if xtr_bottom is not None else None
        self.n_crit: np.ndarray | None = np.atleast_1d(n_crit) if n_crit is not None else None
        self._is_sequence = is_sequence

    def set_defaults(self) -> None:
        default_values = {
            'mach': 0.0,
            'deflection': 0.0,
            'hinge_position': 0.7,
            'xtr_top': 1.0,
            'xtr_bottom': 1.0,
            'n_crit': 9.0
        }
        for attr, default in default_values.items():
            value = getattr(self, attr, None)
            if (value is None) or (len(value) == 0):
                setattr(self, attr, np.atleast_1d(default))
                self._invalidate_cache()

    @property
    def is_alpha_polar(self) -> bool:
        conditions = [
            isinstance(self.reynolds, float) or len(np.array(self.reynolds)) == 1,
            isinstance(self.mach, float) or len(np.array(self.mach)) == 1,
            isinstance(self.deflection, float) or len(np.array(self.deflection)) == 1,
            isinstance(self.hinge_position, float) or len(np.array(self.hinge_position)) == 1,
            isinstance(self.xtr_top, float) or len(np.array(self.xtr_top)) == 1,
            isinstance(self.xtr_bottom, float) or len(np.array(self.xtr_bottom)) == 1,
            isinstance(self.n_crit, float) or len(np.array(self.n_crit)) == 1,
        ]
        return all(conditions)

    @property
    def is_single_point(self):
        return self.is_alpha_polar and (isinstance(self.alpha, float) or len(np.array(self.alpha)) == 1)

    @property
    def is_deflected(self) -> bool:
        return self.deflection is None and self.hinge_position is None

    @classmethod
    def num_params(cls) -> int:
        return 8
