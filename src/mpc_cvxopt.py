import numpy as np
import time
import cvxopt
# import osqp
from scipy import sparse
# import pyqpoases

import logging
import os

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
    def __init__(self, Q=None, R=None):
        """
        Attributes:
            h (int): Time horizon parameter for the MPC.
            dt (float): Time step size.
            x_cmd (np.array): Command vector [Theta, p, Omega, v].
            Q (np.array): State weights [Theta, p, Omega, v, g].
            R (np.array): Control input weights [f1, f2, m1, m2].
        """
        self.initialize_parameters(Q, R)
    
    def initialize_parameters(self, Q, R):
        """
        Initialize parameters for the MPC.
        Args:
            Q (np.array): State weights [Theta, p, Omega, v].
            R (np.array): Control input weights [F_left, M_left] (right uses same params).
        """
        self.h = 10
        self.dt = 0.04
        self.x_cmd = np.array([0, 0, 0, 0, 0, 0.55, 0, 0, 0, 0, 0, 0])  # Command [Theta, p, Omega, v]
        
        if Q is not None:
            if len(Q) != 12:
                raise ValueError("Q must have length 12 for [Theta, p, Omega, v]")
            self.Q = np.append(Q, 1)  # Extend Q with gravity term (always 1)
        else:
            raise ValueError("Q must be provided with length 12 for [Theta, p, Omega, v]")
        
        if R is not None:
            if len(R) != 6:
                raise ValueError("R must have length 6 [F_x, F_y, F_z, M_x, M_y, M_z]")
            # Duplicate R for left and right legs:
            # [F_left_x, F_left_y, F_left_z, F_right_x, F_right_y, F_right_z,
            #  M_left_x, M_left_y, M_left_z, M_right_x, M_right_y, M_right_z]
            R_left_right = np.concatenate([R[:3], R[:3]])
            R_moments = np.concatenate([R[3:], R[3:]])
            self.R = np.concatenate([R_left_right, R_moments])
        else:
            raise ValueError("R must be provided with length 6 [F_x, F_y, F_z, M_x, M_y, M_z]")
        
        self.kv = 0.01  # Velocity gain for foot placement
        self.kp = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 2]]) * 700  # Gains for swing leg control
        self.kd = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) * 3
        self.swingHeight = 0.1
        self.y_offset = 0.04
        self.rl_offsets = np.array([0, 0, 0, 0], dtype=np.float32)  # RL foot offsets [lx, ly, rx, ry]
        
        self.x_fb = np.zeros(12)

    def update_rl_offsets(self, rl_offsets):
        """Update the RL foot offsets [lx, ly, rx, ry]"""
        if len(rl_offsets) != 4:
            raise ValueError("rl_offsets must have length 4 [lx, ly, rx, ry]")
        self.rl_offsets = rl_offsets

    def update_cmd(self, x_cmd, x_fb, frame="world"):
        """
        
        Args:
            x_cmd (np.array): Command vector [vx, vy, vyaw, z].
            x_fb (np.array): Current state feedback [rx, ry, rz, x, y, z, wx, wy, wz, vx, vy, vz].
            frame (str): Frame of reference for the command. Can be "world" or "body". Default is "world".
        
        """
        # Ensure x_cmd is np.array(4)
        if not isinstance(x_cmd, np.ndarray):
            raise ValueError("x_cmd must be a numpy array")
        if x_cmd.shape != (4,):
            raise ValueError("x_cmd must be of shape (4,)")
        
        if frame == "body":
            yaw = x_fb[2]
            self.x_cmd[9] = x_cmd[0] * np.cos(yaw) - x_cmd[1] * np.sin(yaw) # vx
            self.x_cmd[10] = x_cmd[1] * np.cos(yaw) + x_cmd[0] * np.sin(yaw) # vy
        elif frame == "world":
            self.x_cmd[9] = x_cmd[0]
            self.x_cmd[10] = x_cmd[1]
        self.x_cmd[8] = x_cmd[2] # wz
        self.x_cmd[5] = x_cmd[3] # height

        self.x_fb = x_fb
        base_pos = self.x_fb[3:6]
        base_eul = self.x_fb[0:3]

        pos_range = 0.2
        yaw_range = 0.2
        pos_vel_thres = 0.02
        yaw_vel_thres = 0.02
        
        # Update desired positions and yaw based on thresholds
        if abs(base_pos[0] - self.x_cmd[3]) > pos_range or abs(self.x_cmd[9]) > pos_vel_thres:
            self.x_cmd[3] = base_pos[0]
        if abs(base_pos[1] - self.x_cmd[4]) > pos_range or abs(self.x_cmd[10]) > pos_vel_thres:
            self.x_cmd[4] = base_pos[1]
        if abs(base_eul[2] - self.x_cmd[2]) > yaw_range or abs(self.x_cmd[8]) > yaw_vel_thres:
            self.x_cmd[2] = base_eul[2]

    def reset(self, Q, R):
        self.initialize_parameters(Q, R)


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
    def __init__(self, sim_dt=0.001, ctrl_dt=0.02, verbose=False, gait=1, logging=False, mpc_params=None, **kwargs):
        """
        Main controller class that handles MPC and leg control.
        """
        self.instance_id = str(id(self))[-6:]  # Use last 6 digits of instance id
        self.logging = logging
        if self.logging:           
            self.logger = logging.getLogger(f'MPC_{self.instance_id}')
            self.logger.setLevel(logging.INFO)
            log_filename = f'mpc_solver_{self.instance_id}.log'
            if os.path.exists(log_filename):
                os.remove(log_filename)
            fh = logging.FileHandler(log_filename)
            fh.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

        if mpc_params:
            self.mpc = MPC(Q=mpc_params.get("Q"), R=mpc_params.get("R"))
        else:
            self.mpc = MPC()
        self.biped = Biped()

        self.sim_dt = sim_dt
        self.ctrl_dt = ctrl_dt # 0.02
        self.decimation = int(self.ctrl_dt / self.sim_dt)  # Number of simulation steps per control step, 20

        self.verbose = verbose

        cvxopt.solvers.options['show_progress'] = self.verbose
        cvxopt.solvers.options['maxiters'] = 400
        cvxopt.solvers.options['abstol'] = 1e-6
        cvxopt.solvers.options['reltol'] = 1e-6
        cvxopt.solvers.options['feastol'] = 1e-6

        self.foot_l = np.zeros((3, 1))
        self.foot_r = np.zeros((3, 1))

        self.gait = gait

        self.u0 = np.zeros((12, 1))
        self.controls= np.zeros((self.mpc.h, 12))
        self.x_ref = np.tile(np.append(self.mpc.x_cmd, 1), (self.mpc.h, 1)).T
        self.gravity_proj_vec = np.array([0, 0, 0, 0, 0, -self.biped.g])
        self.tau = np.zeros((10, 1))

        self.step_counter = 0

        self.mpc_start_time = time.time()
        self.mpc_end_time = time.time()

    def run_step(self, x_fb, q, qd, gait=None):
        """
        Execute one control step, should run at the same freq as the simulation (1000Hz 0.001s).
        MPC solver runs at 50Hz (0.02s).
        
        Args:
            x_fb: Current state feedback
            t: Current time
            q: Joint positions 
            qd: Joint velocities
            gait: Gait type (0=standing, 1=walking)
            
        Returns:
            tau: Joint torques
        """
        ## 1000Hz here

        t = self.step_counter * self.sim_dt
        if self.verbose:
            print("time: ", t)

        # Get foot positions
        pf_w = self.get_foot_pos_world(x_fb, q)
        foot = pf_w.reshape(-1)

        # Generate contact sequence
        if gait == 0 or gait == 1:
            self.gait = gait
        if self.gait == 1:
            contact = self.get_contact_sequence(t)
        else:
            contact = np.ones((self.mpc.h, 2))

        # self.mpc.x_cmd[5] = 0.55 + 0.05 * np.sin(2 * np.pi * 0.25 * t)

        # Solve MPC, MPC runs once every 0.02 / 0.001 = 20 sim steps
        if self.step_counter % self.decimation == 0:
            ## 50Hz here

            self.mpc_start_time = time.time()
            if self.verbose:
                print(f"Time for everything else: {(self.mpc_start_time - self.mpc_end_time):.3f}s")

            controls = self.solve_mpc(x_fb, t, foot, contact)
            if controls is not None:
                self.controls = controls
            else:
                if self.logging:
                    self.logger.warning("Using previous control solution due to invalid QP result, not updating result.")
                if self.controls is None:
                    self.controls = np.zeros((self.mpc.h, 12))

            self.mpc_end_time = time.time()
            if self.verbose:
                print(f"MPC solving time: {(self.mpc_end_time - self.mpc_start_time):.3f}s")
            self.u0 = self.controls[0, :].reshape(-1, 1) # World frame

        # Generate joint torques
        self.tau = self.low_level_control(x_fb, t, pf_w, q, qd, contact, self.u0)
        if self.verbose:
            print("Torques: \n", self.tau)

        self.step_counter += 1

        # return self.tau, self.states, self.controls, self.x_ref
        return self.tau, self.controls

    def reset(self, mpc_params):
        """
        Reset the controller state.
         inputs
        - Reference trajectories
        - Timing variables
        """
        # Reset MPC class (including x_cmd)
        self.mpc.reset(Q=mpc_params.get("Q"), R=mpc_params.get("R"))

        # Reset step counter
        self.step_counter = 0

        # Reset verbose
        self.verbose = False

        # Reset foot positions
        self.foot_l = np.zeros((3, 1))
        self.foot_r = np.zeros((3, 1))

        # Reset control variables
        self.u0 = np.zeros((12, 1))
        self.states = None
        self.controls = np.zeros((self.mpc.h, 12))
        self.x_ref = np.tile(np.append(self.mpc.x_cmd, 1), (self.mpc.h, 1)).T
        self.gravity_proj_vec = np.array([0, 0, 0, 0, 0, -self.biped.g])
        self.tau = np.zeros((10, 1))

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
        x_ref[:12, 0] = x_fb
        for i in range(6):
            for k in range(0, self.mpc.h):
                if self.mpc.x_cmd[i + 6] != 0:
                    x_ref[i, k] = self.mpc.x_cmd[i] + self.mpc.x_cmd[i + 6] * (k * self.mpc.dt)
                else: # Remaining still
                    x_ref[i, k] = self.mpc.x_cmd[i]
                    # if k < self.mpc.h - 1:
                    #     x_ref[i, k] = self.x_ref[i, k+1] # self.mpc.x_cmd[i]
                    # else:
                    #     x_ref[i, k] = self.x_ref[i, k]
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
        R = eul2rotm(x_fb[0:3])
        rl_offsets = self.mpc.rl_offsets
        y_offset = self.mpc.y_offset
        offset_l = R @ np.array([[rl_offsets[0]],[rl_offsets[1] + y_offset],[0]]) # left
        offset_r = R @ np.array([[rl_offsets[2]],[rl_offsets[3] - y_offset],[0]]) # right

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
            + self.mpc.kv * (x_fb[4] - self.mpc.x_cmd[4])
        )
        foot_des_z = 0
        
        foot_1 = np.array(
            [
                foot_des_x_1 + offset_l[0],
                foot_des_y_1 + offset_l[1],
                foot_des_z + offset_l[2],
                foot_des_x_1 + offset_r[0],
                foot_des_y_1 + offset_r[1],
                foot_des_z + offset_r[2],
            ]
        )
        foot_2 = np.array(
            [
                foot_des_x_2 + offset_l[0],
                foot_des_y_2 + offset_l[1],
                foot_des_z + offset_l[2],
                foot_des_x_2 + offset_r[0],
                foot_des_y_2 + offset_r[1],
                foot_des_z + offset_r[2],
            ]
        )

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

    def set_desired_acc(self, acc, x_fb):
        """
        Args:
            acc: body frame [alpha_x, alpha_y, alpha_z, a_x, a_y, a_z] or only [a_x, a_y, a_z], excluding gravity
        """
        R = eul2rotm(x_fb[0:3])
        
        if len(acc) == 6:
            acc_ang = R @ acc[0:3]
            acc_lin = R @ acc[3:6]
        elif len(acc) == 3:
            acc_ang = np.array([0, 0, 0])
            acc_lin = R @ acc
        else:
            raise ValueError("acc must be of length 3 or 6")
        
        self.gravity_proj_vec = self.gravity_proj_vec = np.concatenate([acc_ang, acc_lin]) + np.array([0, 0, 0, 0, 0, -self.biped.g])

    def get_simplified_dynamics(self, x_ref, foot_ref):
        # Iterate through each step
        # Extract values from x_traj
        yaw = x_ref[2]
        pitch = x_ref[1]
        R = eul2rotm(x_ref[0:3]) # Transform a vector from body frame to world frame
        I = R @ self.biped.I @ R.T # Transform inertia from body to world: I_w = R_b2w @ I_b @ R_b2w.T

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
        R = eul2rotm(x_fb[0:3]) # Transform a vector from body frame to world frame
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
        lt = self.biped.lt - 0.02
        lh = self.biped.lh - 0.02

        # Construct A_LF1
        A_LF1 = np.vstack([
            np.hstack([-lh * np.array([0, 0, 1]) @ R.T, np.zeros(3), np.array([0, 1, 0]) @ R.T, np.zeros(3)]),
            np.hstack([-lt * np.array([0, 0, 1]) @ R.T, np.zeros(3), -np.array([0, 1, 0]) @ R.T, np.zeros(3)]),
            np.hstack([np.zeros(3), -lh * np.array([0, 0, 1]) @ R.T, np.zeros(3), np.array([0, 1, 0]) @ R.T]),
            np.hstack([np.zeros(3), -lt * np.array([0, 0, 1]) @ R.T, np.zeros(3), -np.array([0, 1, 0]) @ R.T]),
            np.hstack([lt * np.array([0, 1, -self.biped.mu]) @ R.T, np.zeros(3), np.array([0, -self.biped.mu, -1]) @ R.T,  np.zeros(3)]),
            np.hstack([np.zeros(3), lt * np.array([0, 1, -self.biped.mu]) @ R.T, np.array([0, -self.biped.mu, -1]) @ R.T,  np.zeros(3)]),
            np.hstack([lt * np.array([0, -1, -self.biped.mu]) @ R.T, np.zeros(3), np.array([0, -self.biped.mu, -1]) @ R.T,  np.zeros(3)]),
            np.hstack([np.zeros(3), lt * np.array([0, -1, -self.biped.mu]) @ R.T, np.array([0, -self.biped.mu, -1]) @ R.T,  np.zeros(3)]),
            np.hstack([lh * np.array([0, 1, -self.biped.mu]) @ R.T, np.zeros(3), np.array([0, self.biped.mu, 1]) @ R.T,  np.zeros(3)]),
            np.hstack([np.zeros(3), lh * np.array([0, 1, -self.biped.mu]) @ R.T, np.array([0, self.biped.mu, 1]) @ R.T,  np.zeros(3)]),
            np.hstack([lh * np.array([0, -1, -self.biped.mu]) @ R.T, np.zeros(3), np.array([0, self.biped.mu, -1]) @ R.T,  np.zeros(3)]),
            np.hstack([np.zeros(3), lh * np.array([0, -1, -self.biped.mu]) @ R.T, np.array([0, self.biped.mu, -1]) @ R.T,  np.zeros(3)]),
        ])

        # Horizon block expansion
        A_LFh = np.kron(np.eye(self.mpc.h), A_LF1)
        padding = np.zeros((12 * self.mpc.h, 13 * self.mpc.h))
        A_LF = np.hstack([padding, A_LFh])

        # Define b_LF
        b_LF = np.zeros((12 * self.mpc.h, 1))

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

        is_optimal, is_useable = self.check_solution_validity(solution)
        if not is_optimal:
            if self.logging:
                self.logger.warning("QP solution may be invalid or not optimal.")
        if not is_useable:
            if self.logging:
                self.logger.error("QP solution is not usable.")
            print("QP solution is not usable.")
            return None

        # Extract states and controls from the solution
        x_opt = np.array(solution['x']).flatten()
        # states = x_opt[:13 * self.mpc.h].reshape((self.mpc.h, 12))
        # controls = x_opt[13 * self.mpc.h:].reshape((self.mpc.h, 12))
        controls = x_opt.reshape((self.mpc.h, 12))

        return controls

    def check_solution_validity(self, solution):
        """Check if the QP solution is valid and optimal
        Returns:
            (is_optimal, is_usable)
        """
        is_optimal, is_useable = True, True
        if solution['status'] == 'optimal':
            pass

        elif solution is None or solution['x'] is None:
            if self.logging:
                self.logger.error("QP solution is None")
            is_optimal, is_useable = False, False

        else:
            is_optimal, is_useable = False, True
            if self.logging:
                self.logger.warning(f"QP solution status: {solution['status']}")

            # Check primal and dual residuals
            primal_infeas = solution.get('primal infeasibility', 0)
            dual_infeas = solution.get('dual infeasibility', 0)
            rel_gap = solution.get('relative gap', 0)
            PRIMAL_TOL = 1e-6
            DUAL_TOL = 1e-6
            GAP_TOL = 1e-6

            if self.logging:
                if primal_infeas > PRIMAL_TOL:
                    self.logger.warning(f"Primal infeasibility ({primal_infeas}) exceeds tolerance ({PRIMAL_TOL})")
                if dual_infeas > DUAL_TOL:
                    self.logger.warning(f"Dual infeasibility ({dual_infeas}) exceeds tolerance ({DUAL_TOL})")
                if rel_gap > GAP_TOL:
                    self.logger.warning(f"QP gap ({rel_gap}) exceeds tolerance ({GAP_TOL})")

        return is_optimal, is_useable

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
        """
        Rotation explanation:
            Here R = eul2rotm(x_fb[0:3]), we need to figure out what exactly R does, and in what frames the rotation is defined.
            When R = array([[ 0.93, -0.37, -0.03],
                            [ 0.37,  0.93,  0.03],
                            [ 0.02, -0.04,  1.  ]]), the robot has euler angles array([-0.04, -0.02,  0.38]),
            and faces about 20 degrees to the left of +x axis, or equivalently, 70 degrees to the right of +y axis
            Let a vector v_w (world frame) = [1, 0, 0], then R @ v_w = [0.93, 0.37, 0.02],
            meaning that R @ v_w actually rotates v_w +20 degrees around +z axis, pointing to the left of +x axis.
            So, if we want to transform a vector from world frame to body frame, we need to use R.T, i.e.
                v_b = R.T @ v_w = [0.93, -0.37, -0.03].
            Conversely, if we want to transform a vector from body frame to world frame, we need to use R, i.e.
                v_w = R @ v_b
        """
        R = eul2rotm(x_fb[0:3]) # Tranform a vector in body frame to world frame
        pf_w = np.zeros((6, 1))
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
        """All in world frame here"""
        R = eul2rotm(x_fb[0:3])
        y_offset = self.mpc.y_offset
        rl_offsets = self.mpc.rl_offsets[0:2] if side == 1 else self.mpc.rl_offsets[2:4]
        offset = R @ np.array([[rl_offsets[0]],[rl_offsets[1] + side * y_offset],[0]])
        foot_des_x = (
            x_fb[3] + x_fb[9] * 1 / 2 * self.mpc.h / 2 * self.mpc.dt
            + self.mpc.kv * (x_fb[3] - self.mpc.x_cmd[3])
        )
        foot_des_y = (
            x_fb[4] + x_fb[10] * 1 / 2 * self.mpc.h / 2 * self.mpc.dt
            + self.mpc.kv * (x_fb[4] - self.mpc.x_cmd[4])
        )

        t = np.remainder(t, self.mpc.dt * self.mpc.h / 2)
        foot_des_z = self.mpc.swingHeight * np.sin(np.pi * t / (self.mpc.dt * self.mpc.h / 2))
        percent = t / (self.mpc.dt * self.mpc.h / 2 )
        if self.verbose: 
            print('percent', percent)
        # if t == 0:
        #     foot_l = np.zeros([3, 1])
        #     foot_r = np.zeros([3, 1])
        if percent < 0.1: 
            if side == 1: # initialize foot position
                self.foot_l = pf_w
            elif side == -1:
                self.foot_r = pf_w
        if side == 1:
            foot_i = self.foot_l
        elif side == -1:
            foot_i = self.foot_r
        foot_des_x = foot_i[0,0] + percent*(foot_des_x - foot_i[0,0])
        foot_des_y = foot_i[1,0] + percent*(foot_des_y - foot_i[1,0])
        foot_des = np.array([[foot_des_x], [foot_des_y], [foot_des_z]]) + offset
        foot_v_des = np.zeros((3,1))
        F_swing = self.mpc.kp @ (foot_des - pf_w) + self.mpc.kd @ (foot_v_des - vf_w)
        return F_swing

    def low_level_control(self, x_fb, t, pf_w, q, qd, contact, u):
        tau = np.zeros((10,1))
        contact = contact[0, 0:2]
        R = eul2rotm(x_fb[0:3]) # Transform a vector from body frame to world frame
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
            u_b = -np.vstack(
                [R.T @ u[3*leg:3*leg+3], R.T @ u[3*leg+6:3*leg+9]]
            )
            tau[5*leg:5*leg+5, :] = Jm.T @ u_b * contact[leg]
            # swing mapping
            tau[5*leg:5*leg+5, :] += Jf.T @ R.T @ F_swing * -(contact[leg] - 1)
            tau[5*leg, :] += (10 * (0 - q0) + 2 * (0 - qd[5*leg])) * -(contact[leg] - 1)

        return tau

    def get_leg_phases(self):
        t = self.step_counter * self.sim_dt
        phase = int(t // self.mpc.dt)
        k = phase % self.mpc.h
        kk = k % 5  # 0, 1, 2, 3, 4

        percentage_phase = np.array([kk] * 2) / 5
        percentage_phase_offset = np.array([0, 0.5])
        percentage_phase = percentage_phase + percentage_phase_offset

        # print(f"phase: {percentage_phase[0]:.2f}%, {percentage_phase[1]:.2f}%", )
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

    mpc_params = {
        'Q': [600, 300, 200, 350, 350, 500, 1, 1, 1, 1, 1, 1], # [rx, ry, rz, x, y, z, wx, wy, wz, vx, vy, vz, gravity(ignored here)]
        'R': [1.0e-5, 1.0e-5, 1.0e-5, 1.0e-4, 1.0e-4, 1.0e-4]
    }

    controller = BipedalLocomotionMPC(verbose=True, mpc_params=mpc_params)

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
