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

    SIM_DT = 0.001
    CTRL_DT = 0.02 # 50Hz
    decimation = int(CTRL_DT / SIM_DT) # number of simulation steps per control step

    # load the yaml file
    conf = yaml.load(open(args.conf_path, "r"), Loader=yaml.FullLoader)
    conf["sim"]["headless"] = args.headless
    conf["sim"]["auto_start_sim"] = True

    # create the simulation object
    sim = mujoco_sim_base.MujocoSimBase(**conf["sim"])
    sim.model.opt.timestep = SIM_DT

    # initialize the simulation
    sim.reset()

    # initialize the controller
    controller = BipedalLocomotionMPC(sim_dt=SIM_DT, ctrl_dt=CTRL_DT, verbose=False)

    vxCommand = 0
    vyCommand = 0
    vyawCommand = 0
    heightCmd = 0.55
    counter = 0
    rl_counter = 0 # Simulates RL env's counter, update once when calling run_step 10 times
    previous_rl_counter = 0

    base_pos_tru = []
    base_tvel_tru = []

    while True:
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

            if rl_counter > previous_rl_counter:
                if rl_counter < 200:
                    vxCommand += 0.001
                elif rl_counter < 400:
                    vxCommand -= 0.001
                elif rl_counter < 600:
                    vyCommand += 0.001
                elif rl_counter < 800:
                    vyCommand -= 0.001
                elif rl_counter < 1000:
                    vyawCommand += 0.002
                elif rl_counter < 1100:
                    vxCommand -= 0.001
                else:
                    pass
                controller.mpc.update_cmd(np.array([vxCommand, vyCommand, vyawCommand, heightCmd]))
                x_ref = controller.get_reference_trajectory(x_fb)
                print('\nrl_counter:', rl_counter)
                print(f"CMD:\tvx = {controller.mpc.x_cmd[9]:.3f}, vy = {controller.mpc.x_cmd[10]:.3f}, vyaw = {controller.mpc.x_cmd[8]:.3f}")
                print(f"REF TRAJ:\n\tx = {x_ref[3, 1]:.3f}\tvx = {x_ref[9, 1]:.3f}\
                        \n\ty = {x_ref[4, 1]:.3f}\tvy = {x_ref[10, 1]:.3f}\
                        \n\tyaw = {x_ref[2, 1]:.3f}\tvyaw = {x_ref[8, 1]:.3f}")
                print(f"CURR TRAJ:\n\tx = {base_pos[0]:.3f}\tvx = {body_tvel[0]:.3f}\
                        \n\ty = {base_pos[1]:.3f}\tvy = {body_tvel[1]:.3f}\
                        \n\tyaw = {base_eul[2]:.3f}\tvyaw = {body_avel[2]:.3f}")

            # Run controller
            tau, controls = controller.run_step(x_fb, q, qd, gait=1)

            # Apply the controll inputs
            sim.data.ctrl[:] = tau.squeeze()

            # steps += 1
            base_pos_tru.append(sim.data.qpos[0:3].copy())
            base_tvel_tru.append(sim.data.qvel[0:3].copy())

            counter += 1
            previous_rl_counter = rl_counter
            rl_counter = counter // 10

            # if controller.step_counter > 5000:
            #     break

        sim.step()
