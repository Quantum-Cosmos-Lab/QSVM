# Classical libraries
import numpy as np
#import scipy
from scipy.linalg import expm
import copy

# Quantum libraries
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector, DensityMatrix, partial_trace
from qiskit.circuit.library.arithmetic.exact_reciprocal import ExactReciprocal
from qiskit.circuit.library import Initialize, UnitaryGate, PhaseEstimation

#from controlled_initialization import controlled_initialization_matrix

class QSVM:
    def __init__(self, n_l=2, t=2*np.pi*3/8, info=False):
        self.n_l = n_l
        self.t = t
        self.b = np.array([1,0]) # for the exemplary demonstration of the HHL
        self.A = np.array([[1,-1/3],[-1/3,1]]) # for the exemplary demonstration of the HHL
        self.n_b = int(np.log2(len(self.b)))
        self.unitary_matrix = expm(1.0j*self.t*self.A)
        self.info = info

    def initialize_b(self,y):
        M = y.shape[0]
        n_M = int(np.ceil(np.log2(M+1)))
        M_theoretical = np.power(2,n_M)-1

        # Add empty data rows, if necessary
        if(M_theoretical > M):
            M_difference = M_theoretical - M
            while(M_difference):
                y = np.concatenate([y,[0]])
                M_difference -= 1

        self.y=y
        self.b = np.concatenate([[0],y])
        self.b = self.b/np.linalg.norm(self.b)
        return(None)

    def preprocess_data(self,data_array):
        M,N = data_array.shape

        n_M = int(np.ceil(np.log2(M+1)))
        n_N = int(np.ceil(np.log2(N+1)))

        M_theoretical = np.power(2,n_M)-1
        N_theoretical = np.power(2,n_N)-1
        
        # Add empty data rows, if necessary
        if(M_theoretical > M):
            M_difference = M_theoretical - M
            while(M_difference):
                data_array = np.r_[data_array, [0*data_array[0]]]
                M_difference -= 1

        # Add empty features, if necessary
        if(N_theoretical > N):
            N_difference = N_theoretical - N
            data_array = np.array([list(x)+N_difference*[0] for x in data_array])

        # Adjust for the further processing
        X = np.c_[np.zeros((data_array.shape[0])),data_array]
        X = np.r_[[np.zeros(data_array.shape[1]+1)], X]
        self.X = X
        return(None)

    def initialize_kernel(self):
        M = self.X.shape[0]
        N = self.X.shape[1]

        data_vector = self.X.flatten()

        norm = np.linalg.norm(data_vector)
        normalized_data = data_vector / norm

        # Determine the number of qubits needed for each register
        num_qubits_i = int(np.ceil(np.log2(M)))  # Qubits for |i⟩ states
        num_qubits_k = int(np.ceil(np.log2(N)))  # Qubits for |k⟩ states

        # Create quantum registers
        i_register = QuantumRegister(num_qubits_i, name='i')
        k_register = QuantumRegister(num_qubits_k, name='k')
        # Create the quantum circuit with the separate registers
        qc = QuantumCircuit(i_register, k_register)
        # Initialize the quantum circuit with the normalized data
        initializer = Initialize(normalized_data)
        qc.append(initializer, i_register[:] + k_register[:])
        rho_chi = DensityMatrix(qc)
        K_hat = partial_trace(rho_chi, list(range(num_qubits_k))).data
        K_hat = np.array(np.real(K_hat), dtype=float)
        self.K_hat = K_hat
        return(None)

    def star_graph_adj_matrix(self,N):
        J = np.zeros((N,N))
        J[0] += 1
        J[:,0] += 1
        J[0,0] = 0
        return(J)
    
    def K_gamma(self, gamma):
        iden = np.identity(self.K_hat.shape[0])
        iden[0,0] = 0
        K_gamma = self.K_hat + (1/gamma)*iden
        return(K_gamma)

    def initialize_F_hat(self, gamma):
        self.gamma = gamma
        # If K not initialized throw an error
        N = self.K_hat.shape[0]
        J = self.star_graph_adj_matrix(N)
        K_gamma = self.K_gamma(self.gamma)
        F = J + K_gamma
        F_hat = F/np.trace(F)
        self.F_hat = F_hat
        return(None)

    def kernel_2_unitary(self, t):
        self.unitary_matrix = expm(1.0j*t*self.F_hat)
        return(None)

    def hhl(self, check_fidelity=True):
        """
        This function creates an HHL circuit with qiskit, execute it and returns the final statevector which is a solution to the linear system.
        The implementation relies heavily on the https://github.com/Classiq/classiq-library/blob/main/tutorials/technology_demonstrations/hhl/hhl.ipynb notebook provided by Classiq.
        """
        self.n_b = int(np.log2(len(self.b)))
        total_q = self.n_b+self.n_l+1
        vector_circuit = QuantumCircuit(self.n_b)
        initi_vec = Initialize(self.b / np.linalg.norm(self.b))

        vector_circuit.append(
            initi_vec, list(range(self.n_b))
        )

        q = QuantumRegister(self.n_b, "q")
        unitary_qc = QuantumCircuit(q)

        unitary_qc.unitary(self.unitary_matrix.tolist(), q)
        qpe_qc = PhaseEstimation(self.n_l, unitary_qc)
        reciprocal_circuit = ExactReciprocal(
            num_state_qubits=self.n_l, scaling=1 / 2**self.n_l
        )
        # Initialise the quantum registers
        qb = QuantumRegister(self.n_b)  # right hand side and solution
        ql = QuantumRegister(self.n_l)  # eigenvalue evaluation qubits
        qf = QuantumRegister(1)  # flag qubits

        hhl_qc = QuantumCircuit(qb, ql, qf)

        # State preparation
        hhl_qc.append(vector_circuit, qb[:])
        # QPE
        hhl_qc.append(qpe_qc, ql[:] + qb[:])
        # Conditioned rotation
        hhl_qc.append(reciprocal_circuit, ql[::-1] + [qf[0]])
        # QPE inverse
        hhl_qc.append(qpe_qc.inverse(), ql[:] + qb[:])
        self.HHL_circ = hhl_qc
        # Print the circuit
        if(self.info): print(hhl_qc.draw())
        # transpile
        #qc_basis = transpile(hhl_qc, backend)
        #tqc = transpile(
        #    hhl_qc,
        #    basis_gates=["u3", "cx"],
        #    optimization_level=transpilation_options["qiskit"],
        #)
        #depth = tqc.depth()
        #cx_counts = tqc.count_ops()["cx"]
        #total_q = tqc.width()

        # Get the output state vector
        statevector = np.array(Statevector(hhl_qc))
        # post_process
        all_entries = [np.binary_repr(k, total_q) for k in range(2**total_q)]
        sol_indices = [
            int(entry, 2)
            for entry in all_entries
            if entry[0] == "1" and entry[1 : self.n_l + 1] == "0" * self.n_l
        ]
        qsol = statevector[sol_indices]# / (1 / 2**n_l)
        #print('unnormalized qsol: ', qsol)
        #print('qsol norm: ',np.linalg.norm(qsol))
        #print('normalized qsol: ', qsol/np.linalg.norm(qsol))

        if(check_fidelity):
            sol_classical = np.linalg.solve(self.F_hat, self.b)
            fidelity = (
                np.abs(
                    np.dot(
                        sol_classical / np.linalg.norm(sol_classical),
                        qsol / np.linalg.norm(qsol),
                )
                )
                ** 2
            )
            if(self.info): print("Solution's fidelity: ", fidelity)
            self.sol_fidelity = fidelity
        self.HHL_solution = qsol
        print(qsol)
        return(None)

    def fit(self, X ,y ,t=1, gamma=1):
        self.preprocess_data(X)
        self.initialize_b(y)
        self.initialize_kernel()
        self.initialize_F_hat(gamma=gamma)
        self.kernel_2_unitary(t=t)
        statevector=self.hhl()
        return(statevector)

    def get_x_tilde_vector(self, x_predict):
        N = self.X.shape[1]-1
        M = self.K_hat.shape[0]-1
        x_tilde_m = np.array([x_predict for _ in range(M)])
        x_tilde_m = np.c_[np.zeros((M,)),x_tilde_m]
        x_tilde_m = np.r_[[np.zeros((N+1,))],x_tilde_m]
        x_tilde_m[0,0] = 1
        self.x_tilde = x_tilde_m.flatten()/np.linalg.norm(x_tilde_m.flatten())
        return(None)

    def get_u_tilde_vector(self):
        u_tilde_m = copy.deepcopy(self.X)
        u_tilde_m[0,0] = 1
        u_tilde_m = np.array([u_tilde_m[i]*self.HHL_solution[i] for i in range(u_tilde_m.shape[0])])
        self.u_tilde = u_tilde_m.flatten()/np.linalg.norm(u_tilde_m.flatten())
        return(None)

    def ancilla_1_probability(self,statevector):
        """Calculate the probability that the most significant bit (ancilla in our convention) is 1."""
        # Get the number of qubits from the size of the statevector
        num_qubits = int(np.log2(len(statevector)))

        # Initialize the probability sum for MSB = 1
        probability_ancilla_1 = 0.0

        # Iterate over the basis states
        for index, amplitude in enumerate(statevector):
            # Convert the index to its binary representation with num_qubits bits
            binary_state = format(index, f'0{num_qubits}b')
            
            # Check if the most significant bit (MSB) is 1
            if binary_state[0] == '1':  # MSB is the first bit in the binary string
                probability_ancilla_1 += np.abs(amplitude) ** 2  # Add the probability amplitude

        return probability_ancilla_1

    def SWAP_test(self):
        CU_u_m = controlled_initialization_matrix(self.u_tilde)
        CU_x_m = controlled_initialization_matrix(self.x_tilde)

        CU_u = UnitaryGate(CU_u_m, label=r'$U_{\tilde{u}}$')
        CU_x = UnitaryGate(CU_x_m, label=r'$U_{\tilde{x}}$')

        n=int(np.log2(len(self.u_tilde)))
        qubits = list(range(n+1))

        qc = QuantumCircuit(n+1)
        qc.h(qubits[-1])
        qc.append(CU_u,qubits)
        qc.x(qubits[-1])
        qc.append(CU_x,qubits)
        qc.h(qubits[-1])
        return(Statevector(qc))

    def predict(self,x):
        self.get_u_tilde_vector()
        self.get_x_tilde_vector(x)
        state = self.SWAP_test()
        P = self.ancilla_1_probability(state)
        if(P<0.5):
            return(1)
        else:
            return(-1)


# Controlled initialization
import numpy as np

def gram_schmidt(vectors):
    """Apply Gram-Schmidt process to a set of vectors."""
    orthonormal_basis = []
    for v in vectors:
        # Subtract projections of v onto all previously found basis vectors
        for b in orthonormal_basis:
            v = v - np.dot(b, v) * b
        # Normalize the vector
        v = v / np.linalg.norm(v)
        orthonormal_basis.append(v)
    return np.array(orthonormal_basis)

def initialization_unitary(first_col):
    """Construct a unitary matrix with a given normalized vector as the first column."""
    # Ensure the first column is normalized
    first_col = first_col / np.linalg.norm(first_col)
    
    # Get the dimension of the vector space (N)
    N = len(first_col)
    
    # Create an identity matrix as the starting point for other vectors
    # These will serve as the basis to which we apply Gram-Schmidt
    identity_matrix = np.eye(N)
    
    # Remove the first column from the identity matrix as we already have the first column
    remaining_cols = identity_matrix[:, 1:]
    
    # Apply Gram-Schmidt process to orthogonalize the rest of the columns
    full_matrix = np.column_stack((first_col, remaining_cols))
    orthonormal_matrix = gram_schmidt(full_matrix.T).T  # Ensure columns are orthogonalized
    
    return orthonormal_matrix

import numpy as np

def controlled_unitary(U):
    """Create block matrix [[I, 0], [0, U]] where I is an identity matrix of size U.shape[0]."""
    N = U.shape[0]  # Size of the unitary matrix U
    I = np.eye(N)   # Identity matrix of size N
    
    # Create the block matrix [[I, 0], [0, U]]
    top_block = np.hstack((I, np.zeros((N, N))))
    bottom_block = np.hstack((np.zeros((N, N)), U))
    block_matrix = np.vstack((top_block, bottom_block))
    
    return block_matrix

def reverse_controlled_unitary(U):
    """Create block matrix [[U, 0], [0, I]] where I is an identity matrix of size U.shape[0]."""
    N = U.shape[0]  # Size of the unitary matrix U
    I = np.eye(N)   # Identity matrix of size N
    
    # Create the block matrix [[U, 0], [0, I]]
    top_block = np.hstack((U, np.zeros((N, N))))
    bottom_block = np.hstack((np.zeros((N, N)), I))
    block_matrix = np.vstack((top_block, bottom_block))
    
    return block_matrix

def controlled_initialization_matrix(v):
    U = initialization_unitary(v)
    CU = controlled_unitary(U)
    return(CU)