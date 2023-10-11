import numpy as np
import sympy as S

class ManipulabilityData:
    def __init__(self, jacobian: S.Matrix):
        self._jacobian = jacobian
        self._velocity_core = self._calculate_velocity_core()
        self._force_core = self._velocity_core.inv()
        self._velocity_measure = self._calculate_velocity_measure()
        self._force_measure = self._calculate_force_measure()
        self._velocity_axes = self._calculate_velocity_core_eigen()
        self._force_axes = self._calculate_force_core_eigen()

    def velocity_measure(self) -> float:
        return self._velocity_measure

    def force_measure(self) -> float:
        return self._force_measure

    def velocity_eigen(self):
        vectors = [value[0] for value in list(self._velocity_axes.values())]
        dimensions = [ S.sqrt(key) for key in self._velocity_axes.keys() ]
        return dimensions, vectors

    def force_eigen(self):
        vectors = [value[0] for value in list(self._force_axes.values())]
        dimensions = [ S.sqrt(key) for key in self._force_axes.keys() ]
        return dimensions, vectors

    def _calculate_velocity_core_eigen(self):
        velocity_eigens = self._velocity_core.eigenvects()
        return { eigen[0] : eigen[2] for eigen in velocity_eigens }

    def _calculate_force_core_eigen(self):
        force_eigens = self._force_core.eigenvects()
        return { eigen[0] : eigen[2] for eigen in force_eigens }

    def _calculate_velocity_measure(self):
        return S.sqrt(S.det(self._velocity_core))

    def _calculate_force_measure(self):
        return S.sqrt(S.det(self._force_core))

    def _calculate_velocity_core(self):
        try:
            jacobian_pinv = self._jacobian.T * (self._jacobian * self._jacobian.T).inv()
            return jacobian_pinv.T * jacobian_pinv
        except:
            print("Singularity discovered: Adding damping parameter to jacobian.")
            regularizer = 1e-3 * S.eye(self._jacobian.shape[1])
            cheat_jacobian = self._jacobian + regularizer
            jacobian_pinv = cheat_jacobian * (cheat_jacobian * cheat_jacobian.T).inv()
            return (jacobian_pinv.T * jacobian_pinv).inv()

