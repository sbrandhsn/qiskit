# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Tests for the Split2QUnitaries transpiler pass.
"""
from math import pi

from test import QiskitTestCase
import numpy as np


from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import Operator
from qiskit.transpiler import PassManager
from qiskit.quantum_info.operators.predicates import matrix_equal
from qiskit.transpiler.passes import Collect2qBlocks, ConsolidateBlocks
from qiskit.providers.basic_provider import BasicSimulator
from qiskit.transpiler.passes.optimization.split_2q_unitaries import Split2QUnitaries


class TestSplit2QUnitaries(QiskitTestCase):
    """
    Tests to verify that splitting two-qubit unitaries into two single-qubit unitaries works correctly.
    """

    def test_splits(self):
        """Test that the kronecker product of matrices is correctly identified by the pass and that the
        global phase is set correctly."""
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.z(1)
        qc.global_phase += 1.2345
        qc_split = QuantumCircuit(2)
        qc_split.append(UnitaryGate(Operator(qc)), [0, 1])

        pm = PassManager()
        pm.append(Collect2qBlocks())
        pm.append(ConsolidateBlocks())
        pm.append(Split2QUnitaries())
        qc_split = pm.run(qc_split)

        self.assertTrue(Operator(qc).equiv(qc_split))
        self.assertTrue(
            matrix_equal(Operator(qc).data, Operator(qc_split).data, ignore_phase=False)
        )

    def test_2q_identity(self):
        """Test that a 2q unitary matching the identity is correctly processed."""
        qc = QuantumCircuit(2)
        qc.id(0)
        qc.id(1)
        qc.global_phase += 1.2345
        qc_split = QuantumCircuit(2)
        qc_split.append(UnitaryGate(Operator(qc)), [0, 1])

        pm = PassManager()
        pm.append(Collect2qBlocks())
        pm.append(ConsolidateBlocks())
        pm.append(Split2QUnitaries())
        qc_split = pm.run(qc_split)

        self.assertTrue(Operator(qc).equiv(qc_split))
        self.assertTrue(
            matrix_equal(Operator(qc).data, Operator(qc_split).data, ignore_phase=False)
        )
        self.assertEqual(qc_split.size(), 2)

    def test_1q_identity(self):
        """Test that a Kronecker product with one identity gate on top is correctly processed."""
        qc = QuantumCircuit(2)
        qc.x(0)
        qc.id(1)
        qc.global_phase += 1.2345
        qc_split = QuantumCircuit(2)
        qc_split.append(UnitaryGate(Operator(qc)), [0, 1])

        pm = PassManager()
        pm.append(Collect2qBlocks())
        pm.append(ConsolidateBlocks())
        pm.append(Split2QUnitaries())
        qc_split = pm.run(qc_split)

        self.assertTrue(Operator(qc).equiv(qc_split))
        self.assertTrue(
            matrix_equal(Operator(qc).data, Operator(qc_split).data, ignore_phase=False)
        )
        self.assertEqual(qc_split.size(), 2)

    def test_1q_identity2(self):
        """Test that a Kronecker product with one identity gate on bottom is correctly processed."""
        qc = QuantumCircuit(2)
        qc.id(0)
        qc.x(1)
        qc.global_phase += 1.2345
        qc_split = QuantumCircuit(2)
        qc_split.append(UnitaryGate(Operator(qc)), [0, 1])

        pm = PassManager()
        pm.append(Collect2qBlocks())
        pm.append(ConsolidateBlocks())
        pm.append(Split2QUnitaries())
        qc_split = pm.run(qc_split)

        self.assertTrue(Operator(qc).equiv(qc_split))
        self.assertTrue(
            matrix_equal(Operator(qc).data, Operator(qc_split).data, ignore_phase=False)
        )
        self.assertEqual(qc_split.size(), 2)

    def test_2_1q(self):
        """Test that a Kronechker product of two X gates is correctly processed."""
        x_mat = np.array([[0, 1], [1, 0]])
        multi_x = np.kron(x_mat, x_mat)
        qr = QuantumRegister(2, "qr")
        backend = BasicSimulator()
        qc = QuantumCircuit(qr)
        qc.unitary(multi_x, qr)
        qct = transpile(qc, backend)

        self.assertTrue(Operator(qc).equiv(qct))
        self.assertTrue(matrix_equal(Operator(qc).data, Operator(qct).data, ignore_phase=False))
        self.assertEqual(qct.size(), 2)

    def test_no_split(self):
        """Test that the pass does not split a non-local two-qubit unitary."""
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.global_phase += 1.2345

        qc_split = QuantumCircuit(2)
        qc_split.append(UnitaryGate(Operator(qc)), [0, 1])

        pm = PassManager()
        pm.append(Collect2qBlocks())
        pm.append(ConsolidateBlocks())
        pm.append(Split2QUnitaries())
        qc_split = pm.run(qc_split)

        self.assertTrue(Operator(qc).equiv(qc_split))
        self.assertTrue(
            matrix_equal(Operator(qc).data, Operator(qc_split).data, ignore_phase=False)
        )
        # either not a unitary gate, or the unitary has been consolidated to a 2q-unitary by another pass
        self.assertTrue(
            all(
                op.name != "unitary" or (op.name == "unitary" and len(op.qubits) > 1)
                for op in qc_split.data
            )
        )

    def test_almost_identity(self):
        """Test that the pass handles QFT correctly."""
        qc = QuantumCircuit(2)
        qc.cp(pi * 2 ** -(26), 0, 1)
        pm = PassManager()
        pm.append(Collect2qBlocks())
        pm.append(ConsolidateBlocks())
        pm.append(Split2QUnitaries(fidelity=1.0 - 1e-9))
        qc_split = pm.run(qc)
        pm = PassManager()
        pm.append(Collect2qBlocks())
        pm.append(ConsolidateBlocks())
        pm.append(Split2QUnitaries())
        qc_split2 = pm.run(qc)
        self.assertEqual(qc_split.num_nonlocal_gates(), 0)
        self.assertEqual(qc_split2.num_nonlocal_gates(), 1)

    def test_split_qft(self):
        """Test that the pass handles QFT correctly."""
        qc = QuantumCircuit(100)
        qc.h(0)
        for i in range(qc.num_qubits - 2, 0, -1):
            qc.cp(pi * 2 ** -(qc.num_qubits - 1 - i), qc.num_qubits - 1, i)
        pm = PassManager()
        pm.append(Collect2qBlocks())
        pm.append(ConsolidateBlocks())
        pm.append(Split2QUnitaries())
        qc_split = pm.run(qc)
        self.assertEqual(26, qc_split.num_nonlocal_gates())