import sys
sys.path.append("./")
from src import mujoco_sim_base
from src.mpc import BipedalLocomotionMPC, MPC, Biped, eul2rotm
from src.transformations import *
import numpy as np
import argparse
import yaml
# from pynput import keyboard


# # keyboard utils
# def on_press(key, steps, controller):
#     global key_pressed
#     key_pressed = True
#     try:
#         print('step:', steps, end=' ')
#         if key == keyboard.Key.up:
#             controller.mpc.x_cmd[9] = 0.5
#             controller.mpc.x_cmd[3] += controller.mpc.x_cmd[9] * SIM_DT 
#             print('mpc.x_cmd[3]:', controller.mpc.x_cmd[3], 'mpc.x_cmd[9]:', controller.mpc.x_cmd[9])
#         elif key == keyboard.Key.down:
#             controller.mpc.x_cmd[9] = -0.5
#             controller.mpc.x_cmd[3] += controller.mpc.x_cmd[9] * SIM_DT
#             print('mpc.x_cmd[3]:', controller.mpc.x_cmd[3], 'mpc.x_cmd[9]:', controller.mpc.x_cmd[9])
#         elif key == keyboard.Key.left:
#             controller.mpc.x_cmd[10] = -0.3
#             controller.mpc.x_cmd[4] += controller.mpc.x_cmd[10] * SIM_DT 
#             print('mpc.x_cmd[4]:', controller.mpc.x_cmd[4], 'mpc.x_cmd[10]:', controller.mpc.x_cmd[10])
#         elif key == keyboard.Key.right:
#             controller.mpc.x_cmd[10] = 0.3
#             controller.mpc.x_cmd[4] += controller.mpc.x_cmd[10] * SIM_DT 
#             print('mpc.x_cmd[4]:', controller.mpc.x_cmd[4], 'mpc.x_cmd[10]:', controller.mpc.x_cmd[10])
#         elif key == keyboard.Key.rshift:
#             controller.gait = 1

#     except AttributeError:
#         print(f'Special key {key} pressed')


# def on_release(key):
#     global key_pressed
#     key_pressed = False
#     if key == keyboard.Key.esc:
#         # Stop listener
#         return False


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
    mpc_params = {
        "Q": [599, 300, 200, 350, 350, 500, 1, 1, 1, 1, 1, 1],
        "R": [1.0e-5, 1.0e-5, 1.0e-5, 1.0e-4, 1.0e-4, 1.0e-4],
    }

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
    controller = BipedalLocomotionMPC(sim_dt=SIM_DT, ctrl_dt=CTRL_DT, verbose=False, mpc_params=mpc_params)

    vxCommand = 0
    vyCommand = 0
    vyawCommand = 0
    heightCmd = 0.55
    counter = 0
    rl_counter = 0 # Simulates RL env's counter, update once when calling run_step 10 times
    previous_rl_counter = 0

    base_pos_tru = []
    base_tvel_tru = []

    # key_pressed = False
    # listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    # listener.start()
    # print('######################## keyboard setup ########################')
    # print('up    - vx = 0.5  and px += vx * dt')
    # print('down  - vx = -0.5 and px += vx * dt')
    # print('left  - vy = -0.3 and py += vy * dt')
    # print('right - vy = 0.3  and py += vy * dt')
    # print('space - pause/unpause')
    # print('key relase and no key pressed - stand at current position')
    # print('#################################################################')

    while True:
        if not sim.viewer_pause:
            base_pos = sim.data.qpos[0:3]
            base_quat = sim.data.qpos[3:7] # wxyz
            base_eul = euler_from_quaternion(base_quat, quat_order='wxyz')
            R = eul2rotm(base_eul) # Transform a vector from body frame to world frame
            body_tvel = sim.data.qvel[0:3]
            body_avel = R @ sim.data.qvel[3:6]
            # body_avel = R @ body_avel

            # joint: l_hip_yaw, l_hip_roll, l_hip_pitch, l_knee, l_ankle, r_hip_yaw, r_hip_roll, r_hip_pitch, r_knee, r_ankle
            q = sim.data.qpos[7:]
            qd = sim.data.qvel[6:]
            x_fb = np.concatenate([base_eul, base_pos, body_avel, body_tvel])

            if rl_counter > previous_rl_counter:
                interval = 200
                if rl_counter == 0:
                    pass
                elif rl_counter < interval:
                    vxCommand += 0.001
                elif rl_counter < 2 * interval:
                    vxCommand -= 0.001
                elif rl_counter < 3 * interval:
                    vyCommand += 0.001
                elif rl_counter < 4 * interval:
                    vyCommand -= 0.001
                elif rl_counter < 5 * interval:
                    vyawCommand += 0.001
                elif rl_counter < 6 * interval:
                    vyawCommand -= 0.001
                elif rl_counter < 7 * interval:
                    vxCommand -= 0.001
                elif rl_counter < 8 * interval:
                    vxCommand += 0.001
                else:
                    pass
                controller.mpc.update_cmd(np.array([vxCommand, vyCommand, vyawCommand, heightCmd]), x_fb)
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
            tau, controls = controller.run_step(x_fb, q, qd)

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
