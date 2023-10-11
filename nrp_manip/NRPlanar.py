import numpy as np
import sympy as S
import sympy.abc as S_sym

class NRPlanar:
    def __init__(self, num_joints: int, rotational_axis: str = 'z'):
        self._joints = [ S.Symbol("q"+str(i)) for i in range(1, num_joints + 1) ]
        self._links = [ S.Symbol("L" + str(i)) for i in range(1, num_joints + 1) ]
        self._link_lengths: list
        if rotational_axis not in { 'x', 'y', 'z' }:
            raise ValueError("NRPlanar: Argument 'rotational_axis' must only be 'x', 'y' or 'z'")
        self._rotational_axis = rotational_axis
        self._calculate_symbolic_forward_kinematics()
        self._sym_jacobian = self._calculate_symbolic_jacobian()
        self._numeric_jacobian: S.Matrix
        self._numeric_fk: S.Matrix

    def init_link_lengths(self, link_lengths: list):
        if len(link_lengths) != len(self._links):
            raise ValueError("NRPlanar.init_link_lengths: Argument 'link_lengths' and NRPlanar._links must have the same size")
        self._xe = self._xe.subs({ symbol : numeric for symbol, numeric in zip(self._links, link_lengths) }) if not isinstance(self._xe, float) else 0.0
        self._ye = self._ye.subs({ symbol : numeric for symbol, numeric in zip(self._links, link_lengths) }) if not isinstance(self._ye, float) else 0.0
        self._ze = self._ze.subs({ symbol : numeric for symbol, numeric in zip(self._links, link_lengths) }) if not isinstance(self._ze, float) else 0.0
        self._sym_jacobian = self._sym_jacobian.subs({ symbol : numeric for symbol, numeric in zip(self._links, link_lengths) })
        self._link_lengths = link_lengths

    def forward_kinematics(self, config: list) -> S.Matrix:
        if len(config) != len(self._joints):
            raise ValueError("NRPlanar.init_link_lengths: Argument 'config' and NRPlanar._joints must have the same size")
        complete_fk = S.Matrix([self._xe, self._ye, self._ze, self._angle])
        self._numeric_fk = complete_fk.subs({ symbol : numeric for symbol, numeric in zip(self._joints, config) })
        return self._numeric_fk

    def numeric_jacobian(self, config: list) -> S.Matrix:
        if len(config) != len(self._joints):
            raise ValueError("NRPlanar.init_link_lengths: Argument 'config' and NRPlanar._joints must have the same size")
        return self._sym_jacobian.subs({ symbol : numeric for symbol, numeric in zip(self._joints, config) })

    def calculate_joint_world_coordinates(self, config: list) -> (list, list):
        if len(config) != len(self._joints):
            raise ValueError("NRPlanar.init_link_lengths: Argument 'config' and NRPlanar._joints must have the same size")
        x_coords = [ sum([ self._link_lengths[j] * S.cos(sum(config[:j+1])) for j in range(0, i+1) ]) for i in range(len(self._joints))]
        y_coords = [ sum([ self._link_lengths[j] * S.sin(sum(config[:j+1])) for j in range(0, i+1) ]) for i in range(len(self._joints))]
        return x_coords, y_coords

    def _calculate_symbolic_forward_kinematics(self):
        if self._rotational_axis == 'z':
            self._xe = self._calculate_aligned_with_plotx()
            self._ye = self._calculate_aligned_with_ploty()
            self._ze = 0.0
        elif self._rotational_axis == 'y':
            self._xe = self._calculate_aligned_with_ploty()
            self._ye = 0.0
            self._ze = self._calculate_aligned_with_plotx()
        else:
            self._xe = 0.0
            self._ye = self._calculate_aligned_with_plotx()
            self._ze = self._calculate_aligned_with_ploty()
        self._angle = self._calculate_angle()

    # If we stand on the positive rotational axis, which axis aligns with our x?
    def _calculate_aligned_with_plotx(self):
        return sum([ self._links[i] * S.cos(sum(self._joints[:i+1])) for i in range(len(self._joints)) ])

    # If we stand on the positive rotational axis, which axis aligns with our y?
    def _calculate_aligned_with_ploty(self):
        return sum([ self._links[i] * S.sin(sum(self._joints[:i+1])) for i in range(len(self._joints)) ])

    def _calculate_angle(self):
        return sum(self._joints)

    def _calculate_symbolic_jacobian(self):
        if self._rotational_axis == 'z':   # x, y
            return S.Matrix([self._xe, self._ye]).jacobian(S.Matrix(self._joints))
        elif self._rotational_axis == 'y': # x, z
            return S.Matrix([self._xe, self._ze]).jacobian(S.Matrix(self._joints))
        else:                              # y, z
            return S.Matrix([self._ye, self._ze]).jacobian(S.Matrix(self._joints))