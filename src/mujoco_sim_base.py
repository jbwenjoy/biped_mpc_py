import mujoco.viewer
import numpy as np
import mujoco
import time

class MujocoSimBase:

    def __init__(
                self, 
                model_path, 
                headless=True,
                viewer_fps=60,
                auto_start_sim=False,
                ):
        # Load the model
        self.model = mujoco.MjModel.from_xml_path(model_path)
        # self.trunk_id = mujoco.mj_name2id(
        #     self.sim.model,
        #     mujoco.mjtObj.mjOBJ_BODY,  # Specify we're looking for a body
        #     'trunk'
        # )
        self.trunk_id = self.obj_name2id('trunk')
        self.world_root_constraint_id = self.obj_name2id('world_root', type='equality')
        self.base_pos_nominal = np.array([0.0, 0.0, 0.55])
        self.data = mujoco.MjData(self.model)
        self.headless = headless
        self.auto_start_sim = auto_start_sim
        self.start_time = time.time()
        # self.viewer_sync_rate = 1.0 / viewer_fps
        self.step_count = 0
        self.frame_skip = int(1000 / viewer_fps)
        
        if self.headless:
            self.step = self.step_headless
        else:
            self.viewer = mujoco.viewer.launch_passive(
                self.model,
                self.data,
                show_left_ui=False,
                show_right_ui=False,
                key_callback=self.viewer_key_callback,
            )
            self.viewer_pause = not self.auto_start_sim  # Set viewer_pause based on auto_start_sim
            self.step = self.step_head

    def viewer_key_callback(self,keycode):
        if chr(keycode) == ' ':
            self.viewer_pause = not self.viewer_pause
        elif chr(keycode) == 'E':
            self.viewer.opt.frame = not self.viewer.opt.frame
        elif chr(keycode) == 'Q':
            self.set_robot_on_ground()

    def step_headless(self):
        mujoco.mj_step(self.model, self.data)

    def reset(self):
        init_qp = np.array(self.model.keyframe('home').qpos)
        mujoco.mj_resetData(self.model,self.data) 
        self.data.qpos[:] = init_qp
        self.step()
        if not self.headless:
            self.viewer.sync()
            self.viewer_pause = not self.auto_start_sim

    def step_head(self):
        if self.viewer.is_running():
            if not self.viewer_pause:
                mujoco.mj_step(self.model, self.data)
                # # 1. Smoother but slower
                # # self.viewer.sync()
                # # 2. Rough but faster
                # if (time.time() - self.start_time) % self.viewer_sync_rate < 1e-3:
                #     self.viewer.sync()
                self.step_count = (self.step_count + 1) % self.frame_skip
                if self.step_count == 0:
                    self.viewer.sync()
        else:
            exit()

    def obj_name2id(self,name,type='body'):
        type = type.upper()
        return mujoco.mj_name2id(
                                    self.model,
                                    getattr(mujoco.mjtObj, 'mjOBJ_'+type), 
                                    name
                                )

    def obj_id2name(self,obj_id,type='body'):
        type = type.upper() 
        return mujoco.mj_id2name(
                                    self.model,
                                    getattr(mujoco.mjtObj, 'mjOBJ_'+type), 
                                    obj_id
                                )

    def get_sensordata_from_id(self,sensor_id):

        start_n = self.model.sensor_adr[sensor_id]

        if sensor_id == self.model.nsensor -1:
            return self.data.sensordata[start_n:]
        else:
            end_n = self.model.sensor_adr[sensor_id+1]
            return self.data.sensordata[start_n:end_n]

    def set_robot_on_ground(self):
        self.model.eq_active0[self.world_root_constraint_id] = 0
        self.data.eq_active[self.world_root_constraint_id] = 0
        self.data.qpos[2] = self.base_pos_nominal[2] + 0.01
        self.data.qvel = 0.0
        # self.step()
        self.viewer.sync()  
        self.viewer_pause = True

    def set_robot_off_ground(self):
        self.data.qpos[:2] = 0.0
        self.data.qpos[2] = 1.0
        self.data.qpos[3] = 1.0
        self.data.qpos[4:] = 0.0
        self.data.qvel = 0.0
        for _ in range(5):
            self.step()
        self.model.eq_active0[self.world_root_constraint_id] = 1
        self.data.eq_active[self.world_root_constraint_id] = 1
        for _ in range(10):
            self.step()
        self.viewer.sync()

    def GetFootContacts(self):
        """
        Return the contact state for each foot by checking active contacts.
        This implementation considers the feet as the bodies 'l_toe' and 'r_toe'.
        
        Returns:
            contacts (list of bool/int): [left_toe_contact, right_toe_contact]
        """
        contacts = [0, 0] # False
        # Get body IDs for the left and right toe bodies.
        l_toe_bodyid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "l_toe")
        r_toe_bodyid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "r_toe")
        # Loop over all active contacts.
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            # Determine the bodies associated with each contact's geoms.
            body1 = self.model.geom_bodyid[contact.geom1]
            body2 = self.model.geom_bodyid[contact.geom2]
            if body1 == l_toe_bodyid or body2 == l_toe_bodyid:
                contacts[0] = 1 # True
            if body1 == r_toe_bodyid or body2 == r_toe_bodyid:
                contacts[1] = 1 # True
        return contacts
