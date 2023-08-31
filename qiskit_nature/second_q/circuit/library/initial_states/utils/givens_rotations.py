# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# /!\ this code has been modified !
# the basic correction is that the output of _prepare_fermionic_gaussian_state_jw
#
#    yield from _prepare_slater_determinant_jw(register, left_unitary.T[list(occupied_orbitals)])
#    yield from reversed(decomposition)
#
# I have corrected it as follows:
#
#    yield from left_decomposition_reversed
#    yield from right_decomposition_reversed
#
# most of the content of _prepare_slater_determinant_jw has been put into 
# _gates_from_transformation_matrix, 
# so that it can be reused into 
# _prepare_fermionic_gaussian_state_jw
#
"""Private helper functions for initial states."""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np
from qiskit import QuantumRegister
from qiskit.circuit import Gate, Qubit
from qiskit.circuit.library import XGate, XXPlusYYGate
from qiskit_nature.utils import apply_matrix_to_slices, givens_matrix
from qiskit_nature.utils.linalg import fermionic_gaussian_decomposition_jw


def _prepare_slater_determinant_jw(
    register: QuantumRegister, transformation_matrix: np.ndarray
) -> Iterable[Tuple[Gate, Tuple[Qubit, ...]]]:
    """Prepare a Slater determinant under the Jordan-Wigner Transform.

    Args:
        register: The register containing the qubits to use
        transformation_matrix: The transformation matrix describing
            the Slater determinant.

    Yields:
        (gate, qubits) pairs describing the operations, where the qubits
        are provided in a tuple
    """
    decomposition_reversed = _gates_from_transformation_matrix(
    register, transformation_matrix)

    yield from decomposition_reversed

def _gates_from_transformation_matrix(
    register: QuantumRegister, transformation_matrix: np.ndarray
) -> Iterable[Tuple[Gate, Tuple[Qubit, ...]]]:
    """Decomposes transformation matrix V as a sequence of Givens rotations
    acting on specific columns.

    Args:
        register: The register containing the qubits to use.
        transformation_matrix: The transformation matrix describing the Slater determinant.

    Yields:
        (gate, qubits) pairs describing the operations in the order they have to appear in the circuit, where the qubits are provided in a tuple.
        Note that the corresponding gates are in the reverse order with respect to the decomposition of V.
    """
    m, n = transformation_matrix.shape

    # set the first n_particles qubits to 1
    for i in range(m):
        yield XGate(), (register[i],)

    # if all orbitals are filled, no further operations are needed
    if m == n:
        return

    current_matrix = transformation_matrix

    # zero out top right corner by rotating rows; this is a no-op
    for j in reversed(range(n - m + 1, n)):
        # Zero out entries in column j
        for i in range(m - n + j):
            # Zero out entry in row i if needed
            if not np.isclose(current_matrix[i, j], 0.0):
                givens_mat = givens_matrix(current_matrix[i + 1, j], current_matrix[i, j])
                current_matrix = apply_matrix_to_slices(current_matrix, givens_mat, [i + 1, i])

    # decompose matrix into Givens rotations
    decomposition: List[Tuple[Gate, Tuple[Qubit, ...]]] = []
    for i in range(m):
        # zero out the columns in row i
        for j in range(n - m + i, i, -1):
            if not np.isclose(current_matrix[i, j], 0.0):
                # compute Givens rotation
                givens_mat = givens_matrix(current_matrix[i, j - 1], current_matrix[i, j])
                theta = np.arccos(np.real(givens_mat[0, 0]))
                phi = np.angle(givens_mat[0, 1])
                # add operations
                decomposition.append(
                    (XXPlusYYGate(2 * theta, phi - np.pi / 2), (register[j], register[j - 1]))
                )
                # update matrix
                current_matrix = apply_matrix_to_slices(
                    current_matrix, givens_mat, [(Ellipsis, j - 1), (Ellipsis, j)]
                )

    yield from reversed(decomposition)
    
def _prepare_fermionic_gaussian_state_jw(  # pylint: disable=invalid-name
    register: QuantumRegister, transformation_matrix: np.ndarray, occupied_orbitals: Sequence[int]
) -> Iterable[Tuple[Gate, Tuple[Qubit, ...]]]:
    """Prepare a fermionic Gaussian state under the Jordan-Wigner Transform.

    Args:
        register: The register containing the qubits to use
        transformation_matrix: The transformation matrix describing the fermionic Gaussian state
        occupied_orbitals: The pseudo-particle orbitals to fill

    Yields:
        (gate, qubits) pairs describing the operations, where the qubits
        are provided in a tuple
    """
    # compute the right decomposition of W as a product of particle-hole and Givens
    # and compute the "left unitary matrix V"
    right_decomposition, left_unitary = fermionic_gaussian_decomposition_jw(
        register, transformation_matrix
    )
    right_decomposition_reversed = reversed(right_decomposition)
    
    # decompose the left unitary matrix (V^T)_{C :}
    # as a product of Givens only.
    left_transformation_matrix = left_unitary.T[list(occupied_orbitals)]
    left_decomposition_reversed = _gates_from_transformation_matrix(register,left_transformation_matrix)
    
    # output gates in reverse order 
    yield from left_decomposition_reversed
    yield from right_decomposition_reversed
  