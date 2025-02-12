import numpy as np
import time
import cvxopt
import osqp
from scipy import sparse
# import pyqpoases

np.set_printoptions(suppress=True, precision=2)

# Junheng initial update 01/06/2025

## definitions:
# States (13,): euler angles, positions, angular velocity(world frame), linear velocity(world frame), 1
# control input (12,): [force and moment] = [f1; f2; m1; m2]

# # Initialize state feedback and parameters
# x_fb = np.array([0, 0, 0, 0, 0, 0.55, 0, 0, 0, 0, 0, 0])  # States: euler angles, positions, angular velocity, linear velocity
# foot = np.array([0,-0.1,0, 0,0.1,0])
# q = np.array([0,0,-np.pi/4,np.pi/2,-np.pi/4, 0,0,-np.pi/4,np.pi/2,-np.pi/4])
# qd = np.zeros((10))
# t = 0
# gait = 0 # standing = 0; walking = 1;
# verbose = False
################## functions #####################
# solvers.options['show_progress'] = verbose

class MPC:
    def __init__(self):
        """
        Attributes:
            h (int): Time horizon parameter for the MPC.
            dt (float): Time step size.
            x_cmd (np.array): Command vector [Theta, p, Omega, v].
            Q (np.array): State weights [Theta, p, Omega, v, g].
            R (np.array): Control input weights [f1, f2, m1, m2].
        """
        self.initialize_parameters()
    
    def initialize_parameters(self):
        self.h = 10
        self.dt = 0.04
        self.x_cmd = np.array([0, 0, 0, 0, 0, 0.55, 0, 0, 0, 0, 0, 0])  # Command [Theta, p, Omega, v]
        self.Q = np.array([600, 300, 10, 150, 350, 500, 1, 1, 1, 1, 1, 1, 1])  # State weights - walking
        self.R = np.array([1, 1, 1, 1, 1, 1, 10, 10, 10, 10, 10, 10]) * 1e-5  # Control input weights
        self.kv = 0.01 # Velocity gain for foot placement
        self.kp = np.array([[1, 0, 0],[0, 1, 0],[0, 0, 1]])*1000 # Gains for swing leg control
        self.kd = np.array([[1, 0, 0],[0, 1, 0],[0, 0, 1]])*5
        self.swingHeight = 0.1
        self.y_offset = 0.04

    def update_cmd(self, x_cmd):
        # Ensure x_cmd is either np.array(4) or np.array(12)
        if not isinstance(x_cmd, np.ndarray):
            raise ValueError("x_cmd must be a numpy array")
        if x_cmd.shape not in [(4,), (12,)]:
            raise ValueError("x_cmd must be of shape (4,) or (12,)")

        # [rx, ry, rz, x, y, z, wx, wy, wz, vx, vy, vz]
        if x_cmd.shape == self.x_cmd.shape:  
            self.x_cmd = x_cmd

        # When using RL outputs, [vx, vy, wz, z]
        elif x_cmd.shape == (4,):
            self.x_cmd[9] = x_cmd[0]
            self.x_cmd[10] = x_cmd[1]
            self.x_cmd[8] = x_cmd[2]
            self.x_cmd[5] = x_cmd[3]

    def reset(self):
        self.initialize_parameters()


class Biped:
    def __init__(self):
        self.m = 12  # Mass
        self.I = np.array([[0.532, 0, 0],
                          [0, 0.5420, 0],
                          [0, 0, 0.0711]])  # Inertia
        self.lt = 0.09  # toe length
        self.lh = 0.05  # heel length
        self.g = 9.81  # Gravity
        self.hip_offset = np.array([-0.005, 0.047, -0.126])
        self.mu = 0.4
        self.f_max = np.array([[500], [500], [500]])
        self.f_min = np.array([[-500], [-500], [0]])
        self.tau_max =  np.array([[33.5], [33.5], [33.5]])
        self.tau_min = -self.tau_max


class BipedalLocomotionMPC:
    def __init__(self, verbose=False):
        """
        Main controller class that handles MPC and leg control.
        """
        self.mpc = MPC()
        self.biped = Biped()
        self.verbose = verbose
        cvxopt.solvers.options['show_progress'] = self.verbose
        self.foot_l = None
        self.foot_r = None

        self.u0 = np.zeros((12, 1))
        self.controls= None
        self.x_ref = np.tile(np.append(self.mpc.x_cmd, 1), (self.mpc.h, 1)).T
        self.gravity_proj_vec = np.array([0, 0, 0, 0, 0, -self.biped.g])
        self.tau = None

        self.step_counter = 0

        self.mpc_start_time = time.time()
        self.mpc_end_time = time.time()

    def run_step(self, x_fb, q, qd, gait=1):
        """
        Execute one control step.
        
        Args:
            x_fb: Current state feedback
            t: Current time
            q: Joint positions 
            qd: Joint velocities
            gait: Gait type (0=standing, 1=walking)
            
        Returns:
            tau: Joint torques
        """
        t = self.step_counter / 1000
        if self.verbose:
            print("time: ", t)

        # Get foot positions
        pf_w = self.get_foot_pos_world(x_fb, q)
        foot = pf_w.reshape(-1)

        # Generate contact sequence
        if gait == 1:
            contact = self.get_contact_sequence(t)
        else:
            contact = np.ones((self.mpc.h, 2))

        # self.mpc.x_cmd[5] = 0.55 + 0.05 * np.sin(2 * np.pi * 0.25 * t)

        # Solve MPC, MPC runs once every 0.04 * 1000 / 10 = 4 iterations
        if np.remainder(self.step_counter, self.mpc.dt * 1000 / 10) == 0:
            self.mpc_start_time = time.time()
            if self.verbose:
                print(f"Time for everything else: {(self.mpc_start_time - self.mpc_end_time):.3f}s")
            self.controls = self.solve_mpc(x_fb, t, foot, contact)
            self.mpc_end_time = time.time()
            if self.verbose:
                print(f"MPC solving time: {(self.mpc_end_time - self.mpc_start_time):.3f}s")
            self.u0 = self.controls[0, :].reshape(-1, 1)

        # Generate joint torques
        self.tau = self.low_level_control(x_fb, t, pf_w, q, qd, contact, self.u0)
        if self.verbose:
            print("Torques: \n", self.tau)

        self.step_counter += 1

        # return self.tau, self.states, self.controls, self.x_ref
        return self.tau, self.controls
    
    def reset(self):
        """
        Reset the controller state.
         inputs
        - Reference trajectories
        - Timing variables
        """
        # Reset MPC class (including x_cmd)
        self.mpc.reset()
        
        # Reset step counter
        self.step_counter = 0

        # Reset verbose
        self.verbose = False

        # Reset foot positions
        self.foot_l = None
        self.foot_r = None

        # Reset control variables
        self.u0 = np.zeros((12, 1))
        self.states = None
        self.controls = None
        self.x_ref = np.tile(np.append(self.mpc.x_cmd, 1), (self.mpc.h, 1)).T
        self.gravity_proj_vec = np.array([0, 0, 0, 0, 0, -self.biped.g])
        self.tau = None

        # Reset timing
        self.mpc_start_time = time.time()
        self.mpc_end_time = time.time()

        return

    def get_contact_sequence(self, t):
        # Default contact sequence
        contact = np.array([
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        ]).T
        phase = int(t // self.mpc.dt)  # Calculate the phase
        k = phase % self.mpc.h  # Remainder of phase divided by ctrl.mpc.h
        contact = contact[k:k+10, :]
        return contact

    def get_reference_trajectory(self, x_fb):
        x_ref = np.tile(np.append(self.mpc.x_cmd, 1), (self.mpc.h, 1)).T
        # x_ref[:12, 0] = x_fb
        for i in range(6):
            for k in range(0, self.mpc.h):
                if self.mpc.x_cmd[i + 6] != 0:
                    x_ref[i, k] = x_fb[i] + self.mpc.x_cmd[i + 6] * (k * self.mpc.dt)
                else: # Remaining still
                    if k < self.mpc.h - 1:
                        x_ref[i, k] = self.x_ref[i, k+1] # self.mpc.x_cmd[i]
                    else:
                        x_ref[i, k] = self.x_ref[i, k]
        return x_ref

    def get_reference_foot_trajectory(self, x_fb, t, foot, contact):
        """
        Get the reference foot trajectory for the next mpc.h steps
        
        Args:
        - x_fb: current state feedback [Theta, p, Omega, v], (12,)
        - t: current time
        - foot: current foot position [lx, ly, lz, rx, ry, rz]
        - contact: current contact sequence, (h, 2)

        Returns:
        - foot_ref: reference foot trajectory for the next mpc.h steps, (6, h)

        Notes:
        - Generates foot landing positions based on desired velocity and current state
        - Maintains constant foot positions during stance phase
        - Interpolates between current and target positions during swing phase
        - Includes lateral offset for stable walking
        """
        foot_des_x_1 = (
            x_fb[3] + x_fb[9] * 1 / 2 * self.mpc.h / 2 * self.mpc.dt
            + self.mpc.kv * (x_fb[3] - self.mpc.x_cmd[3])
        )
        foot_des_x_2 = (
            x_fb[3] + x_fb[9] * 1 / 2 * self.mpc.h * self.mpc.dt
            + self.mpc.kv * (x_fb[3] - self.mpc.x_cmd[3])
        )

        foot_des_y_1 = (
            x_fb[4] + x_fb[10] * 1 / 2 * self.mpc.h / 2 * self.mpc.dt
            + self.mpc.kv * (x_fb[4] - self.mpc.x_cmd[4]) 
        )
        foot_des_y_2 = (
            x_fb[4] + x_fb[10] * 1 / 2 * self.mpc.h * self.mpc.dt
            + self.mpc.kv * (x_fb[4] - self.mpc.x_cmd[4]) - self.mpc.y_offset
        )
        foot_des_z = 0
        foot_1 = np.array([foot_des_x_1, foot_des_y_1 + self.mpc.y_offset, foot_des_z, foot_des_x_1, foot_des_y_1 - self.mpc.y_offset, foot_des_z])
        foot_2 = np.array([foot_des_x_2, foot_des_y_2 + self.mpc.y_offset, foot_des_z, foot_des_x_2, foot_des_y_2 - self.mpc.y_offset, foot_des_z])

        foot = foot.reshape(-1, 1)
        foot_1 = foot_1.reshape(-1, 1)
        foot_2 = foot_2.reshape(-1, 1)

        phase = int(t // self.mpc.dt)
        k = phase % self.mpc.h
        kk = k % 5  # 0, 1, 2, 3, 4
        if np.sum(contact[0, :]) == 1:
            foot_vec = np.tile(foot, (1, 5 - kk))
            foot_1_vec = np.tile(foot_1, (1, 5))
            foot_2_vec = np.tile(foot_2, (1, kk))
            foot_ref = np.concatenate((foot_vec, foot_1_vec, foot_2_vec), axis=1)
        else:
            foot_ref = np.tile(foot, (1, self.mpc.h))

        # foot_ref = np.tile(foot, (1, self.mpc.h)) # TODO not ideal change this
        return foot_ref
    
    def set_desired_acc(self, acc):
        # acc: [alpha_x, alpha_y, alpha_z, a_x, a_y, a_z], excluding gravity
        self.gravity_proj_vec = acc + np.array([0, 0, 0, 0, 0, -self.biped.g])

    def get_simplified_dynamics(self, x_ref, foot_ref):
        # Iterate through each step
        # Extract values from x_traj
        yaw = x_ref[2]
        pitch = x_ref[1]
        R = eul2rotm(x_ref[0:3])
        I = R.T @ self.biped.I @ R

        # Compute Ac matrix
        R_inv = np.linalg.inv(np.array([
            [np.cos(yaw) * np.cos(pitch), -np.sin(yaw), 0],
            [np.sin(yaw) * np.cos(pitch), np.cos(yaw), 0],
            [-np.sin(pitch), 0, 1]
        ]))

        des_ang_acc = self.gravity_proj_vec[0:3].reshape(3, 1)
        des_lin_acc = self.gravity_proj_vec[3:6].reshape(3, 1)  # gravity already included
        Ac = np.block([
            [np.zeros((3, 3)), np.zeros((3, 3)), R_inv @ np.eye(3), np.zeros((3, 3)), np.zeros((3, 1))],
            [np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)), np.eye(3), np.zeros((3, 1))],
            [np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)), des_ang_acc],
            [np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)), des_lin_acc],
            [np.zeros((1, 13))]
        ])

        # Compute Bc matrix
        skew_1 = skew(-x_ref[3:6] + foot_ref[0:3])
        skew_2 = skew(-x_ref[3:6] + foot_ref[3:6])
        Bc = np.block([
            [np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3))],
            [np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3)), np.zeros((3, 3))],
            [np.linalg.solve(I, skew_1), np.linalg.solve(I, skew_2), np.linalg.solve(I, np.eye(3)), np.linalg.solve(I, np.eye(3))],
            [np.eye(3) / self.biped.m, np.eye(3) / self.biped.m, np.zeros((3, 3)), np.zeros((3, 3))],
            [np.zeros((1, 12))]
        ])
        A = Ac * self.mpc.dt + np.eye(13)
        B = Bc * self.mpc. dt
        return A, B

    def solve_mpc(self, x_fb, t, foot, contact):
        self.x_ref = self.get_reference_trajectory(x_fb)
        foot_ref = self.get_reference_foot_trajectory(x_fb, t, foot, contact)
        if self.verbose:  
            print("state reference: \n", self.x_ref)
            print("contact sequence: \n", contact)
            print("foot reference: \n", foot_ref)
        R = eul2rotm(x_fb[0:3])
        # load state matrices for each horizon:
        A_matrices = []
        B_matrices = []
        for k in range(self.mpc.h):
            A, B = self.get_simplified_dynamics(self.x_ref[:, k], foot_ref[:, k])
            A_matrices.append(A)
            B_matrices.append(B)

        y = np.reshape(self.x_ref.T, (13 * self.mpc.h, 1))

        # Aqp
        Aqp = [np.zeros((13, 13)) for _ in range(self.mpc.h)]
        Aqp[0] =  A_matrices[0]
        for i in range(1, self.mpc.h):
            Aqp[i] = np.dot(Aqp[i - 1],  A_matrices[i])
        Aqp = np.vstack(Aqp)

        # Bqp
        Bqp = [[np.zeros((13, 12)) for _ in range(self.mpc.h)] for _ in range(self.mpc.h)]
        for i in range(self.mpc.h):
            Bqp[i][i] = B_matrices[i]
            for j in range(i):
                Bqp[i][j] = np.linalg.matrix_power( A_matrices[i], i - j) @ B_matrices[j]
        for i in range(self.mpc.h - 1):
            for j in range(i + 1, self.mpc.h):
                Bqp[i][j] = np.zeros((13, 12))
        Bqp = np.block(Bqp)
        
        one = np.array([1])
        x_0 = np.concatenate((x_fb, one), axis=0).reshape(-1,1)

        # zero Mx
        Moment_selection = np.array([1, 0, 0])  # Define Moment_selection
        R_foot_R = R  # Replace with actual rotation matrix
        R_foot_L = R  # Replace with actual rotation matrix

        A_M_1 = np.block([
            [np.zeros((1, 3)), np.zeros((1, 3)), Moment_selection @ R_foot_R.T, np.zeros((1, 3))],
            [np.zeros((1, 3)), np.zeros((1, 3)), np.zeros((1, 3)), Moment_selection @ R_foot_L.T]
        ])
        A_M_h = np.kron(np.eye(self.mpc.h), A_M_1)
        padding = np.zeros((2 * self.mpc.h, 13 * self.mpc.h))
        A_M = np.hstack([padding, A_M_h])
        b_M = np.zeros(2 * self.mpc.h)
        Aeq = A_M_h
        beq = b_M.reshape(-1,1)

        # construct inequality constraints:
        # Friction pyramid constraints
        A_mu1 = np.array([
            [1, 0, -self.biped.mu, *[0] * 9],
            [0, 1, -self.biped.mu, *[0] * 9],
            [-1, 0, -self.biped.mu, *[0] * 9],
            [0, -1, -self.biped.mu, *[0] * 9],
            [*[0] * 3, 1, 0, -self.biped.mu, *[0] * 6],
            [*[0] * 3, 0, 1, -self.biped.mu, *[0] * 6],
            [*[0] * 3, -1, 0, -self.biped.mu, *[0] * 6],
            [*[0] * 3, 0, -1, -self.biped.mu, *[0] * 6],
        ])
        A_mu = np.kron(np.eye(self.mpc.h), A_mu1)
        b_mu = np.zeros((8*self.mpc.h, 1))

        # force saturations
        A_f1 = np.vstack([np.eye(12), -np.eye(12)])
        A_f = np.kron(np.eye(self.mpc.h), A_f1)

        b_f = []
        for k in range(self.mpc.h):
            col_k = np.concatenate([
                contact[k, 0] * self.biped.f_max,
                contact[k, 1] * self.biped.f_max,
                contact[k, 0] * self.biped.tau_max,
                contact[k, 1] * self.biped.tau_max,
                contact[k, 0] * -self.biped.f_min,
                contact[k, 1] * -self.biped.f_min,
                contact[k, 0] * -self.biped.tau_min,
                contact[k, 1] * -self.biped.tau_min
            ])
            b_f.append(col_k)
        b_f = np.vstack(b_f)

        # Line-foot constraints (preventing toe/heel lift)
        lt = self.biped.lt - 0.01
        lh = self.biped.lh - 0.02

        # Construct A_LF1
        A_LF1 = np.vstack([
            np.hstack([-lh * np.array([0, 0, 1]) @ R.T, np.zeros(3), np.array([0, 1, 0]) @ R.T, np.zeros(3)]),
            np.hstack([-lt * np.array([0, 0, 1]) @ R.T, np.zeros(3), -np.array([0, 1, 0]) @ R.T, np.zeros(3)]),
            np.hstack([np.zeros(3), -lh * np.array([0, 0, 1]) @ R.T, np.zeros(3), np.array([0, 1, 0]) @ R.T]),
            np.hstack([np.zeros(3), -lt * np.array([0, 0, 1]) @ R.T, np.zeros(3), -np.array([0, 1, 0]) @ R.T]),
        ])

        # Horizon block expansion
        A_LFh = np.kron(np.eye(self.mpc.h), A_LF1)
        padding = np.zeros((4 * self.mpc.h, 13 * self.mpc.h))
        A_LF = np.hstack([padding, A_LFh])

        # Define b_LF
        b_LF = np.zeros((4 * self.mpc.h, 1))

        Aineq = np.vstack([A_mu, A_f, A_LFh])
        bineq = np.vstack([b_mu, b_f, b_LF])

        # MPC->QP math
        L = np.kron(np.eye(self.mpc.h), np.diag(self.mpc.Q))
        K = np.kron(np.eye(self.mpc.h), np.diag(self.mpc.R))
        H = 2 * (Bqp.T @ L @ Bqp + K)
        f = 2 * Bqp.T @ L @ (Aqp @ x_0 - y)

        # Convert to cvxopt format
        H_cvx = cvxopt.matrix(H)
        f_cvx = cvxopt.matrix(f)
        Aeq_cvx = cvxopt.matrix(Aeq)
        beq_cvx = cvxopt.matrix(beq)
        Aqp_cvx = cvxopt.matrix(Aineq)
        bqp_cvx = cvxopt.matrix(bineq)

        # Solve QP using cvxopt
        solution = cvxopt.solvers.qp(H_cvx, f_cvx, G=Aqp_cvx, h=bqp_cvx, A=Aeq_cvx, b=beq_cvx)

        # Extract states and controls from the solution
        x_opt = np.array(solution['x']).flatten()
        # states = x_opt[:13 * self.mpc.h].reshape((self.mpc.h, 12))
        # controls = x_opt[13 * self.mpc.h:].reshape((self.mpc.h, 12))
        controls = x_opt.reshape((self.mpc.h, 12))

        return controls

    @staticmethod
    def get_leg_kinematics(q0, q1, q2, q3, q4, side):
        # Initialize the Jm matrix
        Jm = np.zeros((6, 5))
        sin = np.sin
        cos = np.cos

        # Fill in the matrix entries
        Jm[0, 0] = sin(q0) * (0.04 * sin(q2 + q3 + q4) + 0.22 * sin(q2 + q3) + 0.22 * sin(q2) + 0.0135) + \
                    cos(q0) * (0.015 * side + cos(q1) * (0.018 * side + 0.0025) - sin(q1) * (0.04 * cos(q2 + q3 + q4) + 0.22 * cos(q2 + q3) + 0.22 * cos(q2)))

        Jm[1, 0] = sin(q0) * (0.015 * side + cos(q1) * (0.018 * side + 0.0025) - sin(q1) * (0.04 * cos(q2 + q3 + q4) + 0.22 * cos(q2 + q3) + 0.22 * cos(q2))) - \
                    cos(q0) * (0.04 * sin(q2 + q3 + q4) + 0.22 * sin(q2 + q3) + 0.22 * sin(q2) + 0.0135)

        Jm[2, 0] = 0.0
        Jm[3, 0] = 0.0
        Jm[4, 0] = 0.0
        Jm[5, 0] = 1.0

        Jm[0, 1] = -sin(q0) * (sin(q1) * (0.018 * side + 0.0025) + cos(q1) * (0.04 * cos(q2 + q3 + q4) + 0.22 * cos(q2 + q3) + 0.22 * cos(q2)))
        Jm[1, 1] = cos(q0) * (sin(q1) * (0.018 * side + 0.0025) + cos(q1) * (0.04 * cos(q2 + q3 + q4) + 0.22 * cos(q2 + q3) + 0.22 * cos(q2)))
        Jm[2, 1] = sin(q1) * (0.04 * cos(q2 + q3 + q4) + 0.22 * cos(q2 + q3) + 0.22 * cos(q2)) - cos(q1) * (0.018 * side + 0.0025)
        Jm[3, 1] = cos(q0)
        Jm[4, 1] = sin(q0)
        Jm[5, 1] = 0.0

        Jm[0, 2] = sin(q0) * sin(q1) * (0.04 * sin(q2 + q3 + q4) + 0.22 * sin(q2 + q3) + 0.22 * sin(q2)) - \
                    cos(q0) * (0.04 * cos(q2 + q3 + q4) + 0.22 * cos(q2 + q3) + 0.22 * cos(q2))

        Jm[1, 2] = -sin(q0) * (0.04 * cos(q2 + q3 + q4) + 0.22 * cos(q2 + q3) + 0.22 * cos(q2)) - \
                    cos(q0) * sin(q1) * (0.04 * sin(q2 + q3 + q4) + 0.22 * sin(q2 + q3) + 0.22 * sin(q2))

        Jm[2, 2] = cos(q1) * (0.04 * sin(q2 + q3 + q4) + 0.22 * sin(q2 + q3) + 0.22 * sin(q2))
        Jm[3, 2] = -cos(q1) * sin(q0)
        Jm[4, 2] = cos(q0) * cos(q1)
        Jm[5, 2] = sin(q1)

        Jm[0, 3] = sin(q0) * sin(q1) * (0.04 * sin(q2 + q3 + q4) + 0.22 * sin(q2 + q3)) - \
                    cos(q0) * (0.04 * cos(q2 + q3 + q4) + 0.22 * cos(q2 + q3))

        Jm[1, 3] = -sin(q0) * (0.04 * cos(q2 + q3 + q4) + 0.22 * cos(q2 + q3)) - \
                    cos(q0) * sin(q1) * (0.04 * sin(q2 + q3 + q4) + 0.22 * sin(q2 + q3))

        Jm[2, 3] = cos(q1) * (0.04 * sin(q2 + q3 + q4) + 0.22 * sin(q2 + q3))
        Jm[3, 3] = -cos(q1) * sin(q0)
        Jm[4, 3] = cos(q0) * cos(q1)
        Jm[5, 3] = sin(q1)

        Jm[0, 4] = 0.04 * sin(q2 + q3 + q4) * sin(q0) * sin(q1) - \
                    0.04 * cos(q2 + q3 + q4) * cos(q0)

        Jm[1, 4] = -0.04 * cos(q2 + q3 + q4) * sin(q0) - \
                    0.04 * sin(q2 + q3 + q4) * cos(q0) * sin(q1)

        Jm[2, 4] = 0.04 * sin(q2 + q3 + q4) * cos(q1)
        Jm[3, 4] = -cos(q1) * sin(q0)
        Jm[4, 4] = cos(q0) * cos(q1)
        Jm[5, 4] = sin(q1)

        Jf = Jm[0:3, :]
        return Jm, Jf

    @staticmethod
    def get_foot_pos_body(q0, q1, q2, q3, q4, side):
        # Initialize the pf vector
        pf = np.zeros(3)

        # Fill in the vector entries
        pf[0] = - (3 * np.cos(q0)) / 200 - \
            (9 * np.sin(q4) * (np.cos(q3) * (np.cos(q0) * np.cos(q2) - np.sin(q0) * np.sin(q1) * np.sin(q2)) - 
                                np.sin(q3) * (np.cos(q0) * np.sin(q2) + np.cos(q2) * np.sin(q0) * np.sin(q1)))) / 250 - \
            (11 * np.cos(q0) * np.sin(q2)) / 50 - \
            ((side) * np.sin(q0)) / 50 - \
            (11 * np.cos(q3) * (np.cos(q0) * np.sin(q2) + np.cos(q2) * np.sin(q0) * np.sin(q1))) / 50 - \
            (11 * np.sin(q3) * (np.cos(q0) * np.cos(q2) - np.sin(q0) * np.sin(q1) * np.sin(q2))) / 50 - \
            (9 * np.cos(q4) * (np.cos(q3) * (np.cos(q0) * np.sin(q2) + np.cos(q2) * np.sin(q0) * np.sin(q1)) + 
                                np.sin(q3) * (np.cos(q0) * np.cos(q2) - np.sin(q0) * np.sin(q1) * np.sin(q2)))) / 250 - \
            (23 * np.cos(q1) * (side) * np.sin(q0)) / 1000 - \
            (11 * np.cos(q2) * np.sin(q0) * np.sin(q1)) / 50

        pf[1] = (np.cos(q0) * (side)) / 50 - \
            (9 * np.sin(q4) * (np.cos(q3) * (np.cos(q2) * np.sin(q0) + np.cos(q0) * np.sin(q1) * np.sin(q2)) - 
                                np.sin(q3) * (np.sin(q0) * np.sin(q2) - np.cos(q0) * np.cos(q2) * np.sin(q1)))) / 250 - \
            (3 * np.sin(q0)) / 200 - \
            (11 * np.sin(q0) * np.sin(q2)) / 50 - \
            (11 * np.cos(q3) * (np.sin(q0) * np.sin(q2) - np.cos(q0) * np.cos(q2) * np.sin(q1))) / 50 - \
            (11 * np.sin(q3) * (np.cos(q2) * np.sin(q0) + np.cos(q0) * np.sin(q1) * np.sin(q2))) / 50 - \
            (9 * np.cos(q4) * (np.cos(q3) * (np.sin(q0) * np.sin(q2) - np.cos(q0) * np.cos(q2) * np.sin(q1)) + 
                                np.sin(q3) * (np.cos(q2) * np.sin(q0) + np.cos(q0) * np.sin(q1) * np.sin(q2)))) / 250 + \
            (23 * np.cos(q0) * np.cos(q1) * (side)) / 1000 + \
            (11 * np.cos(q0) * np.cos(q2) * np.sin(q1)) / 50

        pf[2] = (23 * (side) * np.sin(q1)) / 1000 - \
            (11 * np.cos(q1) * np.cos(q2)) / 50 - \
            (9 * np.cos(q4) * (np.cos(q1) * np.cos(q2) * np.cos(q3) - np.cos(q1) * np.sin(q2) * np.sin(q3))) / 250 + \
            (9 * np.sin(q4) * (np.cos(q1) * np.cos(q2) * np.sin(q3) + np.cos(q1) * np.cos(q3) * np.sin(q2))) / 250 - \
            (11 * np.cos(q1) * np.cos(q2) * np.cos(q3)) / 50 + \
            (11 * np.cos(q1) * np.sin(q2) * np.sin(q3)) / 50 - \
            3.0 / 50.0    

        return pf

    def get_foot_pos_world(self, x_fb, q):
        R = eul2rotm(x_fb[0:3])
        pf_w = np.zeros((6,1))
        for leg in range(2):
            q0 = q[5*leg+0]
            q1 = q[5*leg+1]
            q2 = q[5*leg+2]
            q3 = q[5*leg+3]
            q4 = q[5*leg+4]
            if leg == 0:
                side = 1
            else:
                side = -1
            pf_b = self.get_foot_pos_body(q0, q1, q2, q3, q4, side)
            pf_b = pf_b.reshape(-1,1)
            hip_offset = np.array([[self.biped.hip_offset[0]],  [side* self.biped.hip_offset[1]], [self.biped.hip_offset[2]]])
            p_c = x_fb[3:6].reshape(-1,1)
            pf_w[0+3*leg : 3+3*leg] = p_c + R@(pf_b+hip_offset)
        return pf_w

    def swing_leg_control(self, x_fb, t, pf_w, vf_w, side):
        y_offset = self.mpc.y_offset
        foot_des_x = (
            x_fb[3] + x_fb[9] * 1 / 2 * self.mpc.h / 2 * self.mpc.dt
            + self.mpc.kv * (x_fb[3] - self.mpc.x_cmd[3])
        )
        foot_des_y = (
            x_fb[4] + x_fb[10] * 1 / 2 * self.mpc.h / 2 * self.mpc.dt
            + self.mpc.kv * (x_fb[4] - self.mpc.x_cmd[4]) + y_offset*side
        )
        t = np.remainder(t, self.mpc.dt * self.mpc.h / 2)
        foot_des_z = self.mpc.swingHeight * np.sin(np.pi * t / (self.mpc.dt * self.mpc.h / 2))
        percent = t / (self.mpc.dt * self.mpc.h / 2 )
        if self.verbose: 
            print('percent', percent)
        # if t == 0:
        #     foot_l = np.zeros([3, 1])
        #     foot_r = np.zeros([3, 1])
        if percent == 0.0: 
            if side == 1: # initialize foot position
                self.foot_l = pf_w
            elif side == -1:
                self.foot_r = pf_w
        if side == 1:
            foot_i = self.foot_l
        elif side == -1:
            foot_i = self.foot_r
        if self.verbose: print('foot_i',foot_i)
        foot_des_x = foot_i[0,0] + percent*(foot_des_x - foot_i[0,0])
        foot_des_y = foot_i[1,0] + percent*(foot_des_y - foot_i[1,0])
        if self.verbose: print('foot_i', foot_i)
        foot_des = np.array([[foot_des_x],[foot_des_y],[foot_des_z]])
        foot_v_des = np.zeros((3,1))
        F_swing = self.mpc.kp@(foot_des - pf_w) + self.mpc.kd@(foot_v_des - vf_w)
        return F_swing

    def low_level_control(self, x_fb, t, pf_w, q, qd, contact, u):
        tau = np.zeros((10,1))
        contact = contact[0, 0:2]
        R = eul2rotm(x_fb[0:3])
        for leg in range(2):
            q0 = q[5*leg+0]
            q1 = q[5*leg+1]
            q2 = q[5*leg+2]
            q3 = q[5*leg+3]
            q4 = q[5*leg+4]
            if leg == 0:
                side = 1
            else:
                side = -1
            # get Jacobians
            Jm, Jf = self.get_leg_kinematics(q0, q1, q2, q3, q4, side)
            # foot velocity in world
            vf_w = R @ Jf @ qd[5*leg:5*leg+5].reshape(-1, 1)
            # swing let force
            F_swing = self.swing_leg_control(x_fb, t, pf_w[3*leg:3*leg+3], vf_w, side)
            # stance mapping
            u_w = -np.vstack(
                [R.T @ u[3*leg:3*leg+3], R.T @ u[3*leg+6:3*leg+9]]
            )
            tau[5*leg:5*leg+5, :] = Jm.T @ u_w * contact[leg]
            # swing mapping
            tau[5*leg:5*leg+5, :] += Jf.T @ R.T @ F_swing * -(contact[leg] - 1)
            tau[5*leg, :] = 30 * (0 - q0) + 1 * (0 - qd[5 * leg])

        return tau

    def get_leg_phases(self):
        t = self.step_counter / 1000
        phase = int(t // self.mpc.dt)
        k = phase % self.mpc.h
        kk = k % 5  # 0, 1, 2, 3, 4

        percentage_phase = np.array([kk] * 2) / 5
        percentage_phase_offset = np.array([0, 0.5])
        percentage_phase = percentage_phase + percentage_phase_offset

        return percentage_phase


def eul2rotm(eul):
    """
    Example 3-2-1 intrinsic Euler angles: eul = [roll, pitch, yaw].
    In MATLAB, you used eul2rotm(flip(eul')), which might be [yaw, pitch, roll].
    This is a placeholder. You must adjust for your actual rotation convention.
    """
    # If you want the exact approach:
    #   eul is 3x1 = [r, p, y], but "flip(eul')" => [y, p, r]
    #   Possibly do something with SciPy:
    #       from scipy.spatial.transform import Rotation as R
    #       R.from_euler('zyx', [yaw, pitch, roll]).as_matrix()
    #
    # For demonstration, let's say your eul = [roll, pitch, yaw]
    cr, cp, cy = np.cos(eul)
    sr, sp, sy = np.sin(eul)

    # Z-Y-X rotation (roll around X, pitch around Y, yaw around Z)
    Rz = np.array([[ cy, -sy,  0 ],
                [ sy,  cy,  0 ],
                [  0,   0,  1 ]])
    Ry = np.array([[ cp,  0,  sp ],
                [  0,  1,   0 ],
                [-sp,  0,  cp ]])
    Rx = np.array([[ 1,  0,   0 ],
                [ 0, cr, -sr ],
                [ 0, sr,  cr ]])
    # Combined: Rz * Ry * Rx
    return Rz @ Ry @ Rx

def skew(v):
    """Skew-symmetric matrix of a 3D vector v."""
    return np.array([
        [    0, -v[2],  v[1]],
        [ v[2],     0, -v[0]],
        [-v[1],  v[0],    0]
    ])

############################## Main Script ###################################

if __name__ == "__main__":
    # Initialize state feedback and parameters
    x_fb = np.array([0, 0, 0, 0, 0, 0.55, 0, 0, 0, 0, 0, 0])  # States: euler angles, positions, angular velocity, linear velocity
    foot = np.array([0,-0.1,0, 0,0.1,0])
    q = np.array([0,0,-np.pi/4,np.pi/2,-np.pi/4, 0,0,-np.pi/4,np.pi/2,-np.pi/4])
    qd = np.zeros((10))
    t = 0
    gait = 1 # standing = 0; walking = 1;

    controller = BipedalLocomotionMPC(verbose=True)

    # # forward kinematics
    # pf_w = controller.get_foot_pos_world(x_fb, q)
    # foot = pf_w.reshape(-1)

    # # contact sequence generation
    # if gait == 1:
    #     contact = controller.get_contact_sequence(t)
    # elif gait == 0:
    #     contact = np.ones((controller.mpc.h, 2))

    # # run MPC
    # start_time = time.time()
    # states, controls = controller.solve_mpc(x_fb, t, foot, contact)
    # end_time = time.time()
    # print(f"MPC Function execution time: {end_time - start_time} seconds")
    # print("States: \n", states)
    # print("Controls: \n", controls)

    # # low level force-to-torque
    # u0 = controls[0, :].reshape(-1,1)
    # tau = controller.low_level_control(x_fb, t, pf_w, q, qd, contact, u0)
    # print("Torques: \n", tau)

    tau, controls = controller.run_step(x_fb, q, qd, gait=gait)
    print("Controls: \n", controls)
    print("Torques: \n", tau)
    