import sys
sys.path.append("./")
from src import mujoco_sim_base
from src.bipedalLocomotionMPC import BipedalLocomotionMPC, MPC, Biped
from src.transformations import *
import numpy as np
import argparse
import yaml

# TO:CHECK
# 1. joint zeros and axis direction
# 2. control desimation and simualtion dt
# 3. order of tau from MPC
if __name__ == "__main__":

    argparser = argparse.ArgumentParser(description="Run the simulation")
    argparser.add_argument(
        "--conf_path",
        type=str,
        help="Path to the configuration file",
        default="config/default.yaml",
    )
    argparser.add_argument(
        "--headless",
        default=False,
        action="store_true",
        help="Run the simulation in headless mode",
    )
    args = argparser.parse_args()

    # load the yaml file
    conf = yaml.load(open(args.conf_path, "r"), Loader=yaml.FullLoader)
    conf["sim"]["headless"] = args.headless
    conf["sim"]["auto_start_sim"] = True

    # create the simulation object
    sim = mujoco_sim_base.MujocoSimBase(**conf["sim"])

    # initialize the simulation
    sim.reset()
    # steps = 0
    max_steps = 6000
    print("max_steps:", max_steps)
    # initialize the controller
    mpc = MPC()
    biped = Biped()
    u0 = np.zeros([12, 1])

    # initialize the controller
    controller = BipedalLocomotionMPC()

    steps = 0
    t = 0

    gait = 1  # standing = 0; walking = 1;
    verbose = False
    # global foot_des_i
    global foot_l
    global foot_r

    base_pos_tru = []
    base_pos_ref = []
    base_tvel_tru = []
    base_tvel_ref = []

    while True:
        # pretty_print_low_cmd(cmd)
        if not sim.viewer_pause:

            base_pos = sim.data.qpos[0:3]
            base_quat = sim.data.qpos[3:7]
            base_eul = quat_to_euler(base_quat)
            body_tvel = sim.data.qvel[0:3]
            body_avel = sim.data.qvel[3:6]

            # joint: l_hip_yaw, l_hip_roll, l_hip_pitch, l_knee, l_ankle, r_hip_yaw, r_hip_roll, r_hip_pitch, r_knee, r_ankle
            q = sim.data.qpos[7:]
            qd = sim.data.qvel[6:]
            x_fb = np.concatenate([base_eul, base_pos, body_avel, body_tvel])

            # Run controller
            tau, controls = controller.run_step(x_fb, q, qd, gait=1)

            # Apply the controll inputs
            sim.data.ctrl[:] = tau.squeeze()

            # steps += 1
            base_pos_tru.append(sim.data.qpos[0:3].copy())
            base_tvel_tru.append(sim.data.qvel[0:3].copy())

            if controller.step_counter > max_steps:
                break

        sim.step()
