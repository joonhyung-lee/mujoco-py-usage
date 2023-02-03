import os,time,cv2,glfw,mujoco_py,math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from screeninfo import get_monitors # get monitor size
from util import r2w,trim_scale,quat2r,rpy2r,pr2t
import apriltag

DEFAULT_SIZE = 500

# MuJoCo Parser class
class MuJoCoParserClass(object):
    def __init__(self,
                 name     = 'Robot',
                 xml_path = ''
                ):
        """
            Initialize MuJoCo parser
        """
        self.name        = name
        self.xml_path    = xml_path
        self.cwd         = os.getcwd()
        # Simulation
        self.tick         = 0
        self.VIEWER_EXIST = False
        # Parse the xml file
        self._parse_xml()
        # Reset
        self.reset()

        # camera variables
        self.cam_matrix = None
        self.azimuth    = None
        self.elevation  = None
        self.distance   = None
        self.lookat     = np.zeros(3)
        self._viewers = {}

        self.render_width = 1500
        self.render_height = 1000

    def _parse_xml(self):
        """
            Parse an xml file
        """
        # Basic MuJoCo model and sim
        self.full_xml_path = os.path.abspath(os.path.join(self.cwd,self.xml_path))
        self.model         = mujoco_py.load_model_from_path(self.full_xml_path)
        self.sim           = mujoco_py.MjSim(self.model)
        # Parse model information
        self.dt              = self.sim.model.opt.timestep 
        self.HZ              = int(1/self.dt)
        self.n_body          = self.model.nbody
        self.body_names      = list(self.sim.model.body_names)
        self.n_joint         = self.model.njnt
        self.joint_idxs      = np.arange(0,self.n_joint,1)
        self.joint_names     = [self.sim.model.joint_id2name(x) for x in range(self.n_joint)]
        self.joint_types     = self.sim.model.jnt_type # 0:free, 1:ball, 2:slide, 3:hinge
        self.joint_range     = self.sim.model.jnt_range
        self.actuator_names  = list(self.sim.model.actuator_names)
        self.n_actuator      = len(self.actuator_names)
        self.torque_range    = self.sim.model.actuator_ctrlrange
        self.rev_joint_idxs  = np.where(self.joint_types==3)[0].astype(np.int32) # revolute joint indices
        self.rev_joint_names = [self.joint_names[x] for x in self.rev_joint_idxs]
        self.n_rev_joint     = len(self.rev_joint_idxs)
        self.rev_qvel_idxs   = [self.sim.model.get_joint_qvel_addr(x) for x in self.rev_joint_names]
        self.pri_joint_idxs  = np.where(self.joint_types==2)[0].astype(np.int32) # prismatic joint indices
        self.pri_joint_names = [self.joint_names[x] for x in self.pri_joint_idxs]
        self.n_pri_joint     = len(self.pri_joint_idxs)
        self.geom_names      = list(self.sim.model.geom_names)
        self.n_geom          = len(self.geom_names)

        # # of robot joints
        self.n_robot_joint  = len(get_env_body_names(self))
        self.n_joint_idxs   = np.array([i for i in range(self.n_robot_joint)], dtype="int32")

    def print_env_info(self):
        """
            Print env info
        """
        print ("[%s] Instantiated from [%s]"%(self.name,self.full_xml_path))
        print ("- Simulation timestep is [%.4f]sec and frequency is [%d]HZ"%(self.dt,self.HZ))
        print ("- [%s] has [%d] bodies"%(self.name,self.n_body))
        for b_idx in range(self.n_body):
            body_name  = self.body_names[b_idx]
            print (" [%02d] body name:[%s]"%(b_idx,body_name))
        print ("- [%s] has [%d] joints"%(self.name,self.n_joint))
        for j_idx in range(self.n_joint):
            joint_name = self.joint_names[j_idx]
            joint_type = self.joint_types[j_idx]
            if joint_type == 0:
                joint_type_str = 'free'
            elif joint_type == 1:
                joint_type_str = 'ball'
            elif joint_type == 2:
                joint_type_str = 'prismatic'
            elif joint_type == 3:
                joint_type_str = 'revolute'
            else:
                joint_type_str = 'unknown'
            print (" [%02d] name:[%s] type:[%s] joint range:[%.2f to %.2f]"%
                (j_idx,joint_name,joint_type_str,self.joint_range[j_idx,0],self.joint_range[j_idx,1]))
        print ("- [%s] has [%d] revolute joints"%(self.name,self.n_rev_joint))
        for j_idx in range(self.n_rev_joint):
            rev_joint_idx  = self.rev_joint_idxs[j_idx]
            rev_joint_name = self.rev_joint_names[j_idx]
            print (" [%02d] joint index:[%d] and name:[%s]"%(j_idx,rev_joint_idx,rev_joint_name))
        print  ("- [%s] has [%d] actuators"%(self.name,self.n_actuator))
        for a_idx in range(self.n_actuator):
            actuator_name = self.actuator_names[a_idx]
            print (" [%02d] actuator name:[%s] torque range:[%.2f to %.2f]"%
            (a_idx,actuator_name,self.torque_range[a_idx,0],self.torque_range[a_idx,1]))
        print  ("- [%s] has [%d] geometries"%(self.name,self.n_geom))
        for g_idx in range(self.n_geom):
            geom_name = self.geom_names[g_idx]
            print (" [%02d] geometry name:[%s]"%(g_idx,geom_name))
            
    def plot_scene(self,
                   figsize       = (12,8),
                   render_w      = None,
                   render_h      = None,
                   render_expand = 1.0,
                   title_str     = None,
                   title_fs      = 10,
                   RETURN_IMG    = False
                    ):
        """
            Plot scene
        """
        if (render_w is None) and (render_h is None):
            # default render size matches with actual window
            render_w = self.viwer_width*render_expand
            render_h = self.viwer_height*render_expand
        for _ in range(10):
            img = self.viewer.read_pixels(width=render_w,height=render_h,depth=False)
        img = cv2.flip(cv2.rotate(img,cv2.ROTATE_180),1) # 0:up<->down, 1:left<->right
        if RETURN_IMG: # return RGB image
            return img
        else: # plot image
            plt.figure(figsize=figsize)
            plt.imshow(img)
            if title_str is not None:
                plt.title(title_str,fontsize=title_fs)
            plt.axis('off')
            plt.show()

    def viewer_setup(self):
        """
        This method is called when the viewer is initialized.
        Optionally implement this method, if you need to tinker with camera position
        and so forth.
        """
        pass

    # -----------------------------
    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == "human":
                self.viewer = mujoco_py.MjViewer(self.sim)
            elif mode == "rgb_array" or mode == "depth_array":
                self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, -1)

            self.viewer_setup()
            self._viewers[mode] = self.viewer
        return self.viewer

    def render_scene(
        self,
        mode="human",
        cam_infos=None,
        width=DEFAULT_SIZE,
        height=DEFAULT_SIZE,
        depth_toggle=None,
        camera_id=None,
        camera_name=None,
    ):
        """
            Render a scene that camera is now see-ing.
            And applying Camera pose: cam_infos
        """
        if mode == "rgb_array" or mode == "depth_array":
            if camera_id is not None and camera_name is not None:
                raise ValueError(
                    "Both `camera_id` and `camera_name` cannot be"
                    " specified at the same time."
                )

            no_camera_specified = camera_name is None and camera_id is None
            if no_camera_specified:
                camera_name = "track"

            if camera_id is None and camera_name in self.model._camera_name2id:
                camera_id = self.model.camera_name2id(camera_name)

            self._get_viewer(mode).render(width, height, camera_id=camera_id)

        # Viewer setting
        if cam_infos is not None:
            if cam_infos["cam_azimuth"] is not None:
                self.viewer.cam.azimuth = cam_infos["cam_azimuth"]
            if cam_infos["cam_distance"] is not None:
                self.viewer.cam.distance = cam_infos["cam_distance"]
            if cam_infos["cam_elevation"] is not None:
                self.viewer.cam.elevation = cam_infos["cam_elevation"]
            if cam_infos["cam_lookat"] is not None:
                self.viewer.cam.lookat[0] = cam_infos["cam_lookat"][0]
                self.viewer.cam.lookat[1] = cam_infos["cam_lookat"][1]
                self.viewer.cam.lookat[2] = cam_infos["cam_lookat"][2]

            self.cam_infos = cam_infos # {"cam_distance":self.cam_distance, "cam_azimuth":self.cam_azimuth, "cam_elevation":self.cam_elevation, "cam_lookat":self.lookat}

        for _ in range(10):     # for updating scene bug
            img = self.viewer.read_pixels(width=width,height=height,depth=depth_toggle)

        if depth_toggle:
            img = cv2.flip(cv2.rotate(img[1],cv2.ROTATE_180),1)     # 0:up<->down, 1:left<->right
        else:
            img = cv2.flip(cv2.rotate(img,cv2.ROTATE_180),1)        # 0:up<->down, 1:left<->right

        return img

    def set_cam_infos(self, 
                    cam_distance=None,
                    cam_azimuth=None,
                    cam_elevation=None,
                    cam_lookat=None):
        """
            Set camera inforamtions (just setting)
        """
        # Viewer setting
        if cam_distance is not None:
            self.distance = cam_distance
        if cam_azimuth is not None:
            self.azimuth = cam_azimuth
        if cam_elevation is not None:
            self.elevation = cam_elevation
        if cam_lookat is not None:
            self.lookat = np.array([cam_lookat[i] for i in range(3)])
            # self.lookat[0] = cam_lookat[0]
            # self.lookat[1] = cam_lookat[1]
            # self.lookat[2] = cam_lookat[2]

        self.cam_infos = {"cam_distance":self.distance, "cam_azimuth":self.azimuth, "cam_elevation":self.elevation, "cam_lookat":self.lookat}

        return self.cam_infos

    def get_cam_infos(self):
        """
            Get Camera informations about (distance, azimuth, elevation, lookat)
        """
        self.cam_infos = {"cam_distance":self.distance, "cam_azimuth":self.azimuth, "cam_elevation":self.elevation, "cam_lookat":self.lookat}

        return self.cam_infos

    def depth_2_meters(self, depth_image):
        """
        Converts the depth array delivered by MuJoCo (values between 0 and 1) into actual m values.

        Args:
            depth: The depth array to be converted.
        """

        extend = self.model.stat.extent
        near = self.model.vis.map.znear * extend
        far = self.model.vis.map.zfar * extend
        depth_scale = (1-near/far)
        depth_image = [i * depth_scale for i in depth_image]
        depth_image = [near / (1-i) for i in depth_image]

        return depth_image

    def camera_matrix_and_pose(self, width, height, camera_name=None):
        """
        Initializes all camera parameters that only need to be calculated once.
        """

        no_camera_specified = camera_name is None
        if no_camera_specified:
            camera_name = "track"

        cam_id = self.model.camera_name2id(camera_name)
        # Get field of view, default value is 45.
        fovy = self.model.cam_fovy[cam_id]
        # Calculate focal length
        f = 0.5 * height / np.tan(fovy * np.pi / 360)
        # Construct camera matrix
        self.cam_matrix = np.array(((f, 0, width / 2), (0, f, height / 2), (0, 0, 1)))
        # Rotation of camera in world coordinates
        self.cam_rot_mat = self.model.cam_mat0[cam_id]
        self.cam_rot_mat = np.reshape(self.cam_rot_mat, (3, 3))
        # Position of camera in world coordinates
        self.cam_pos = self.model.cam_pos0[cam_id]

        return self.cam_matrix, self.cam_rot_mat, self.cam_pos

    def reset(self,RESET_GLFW=False):
        """
             Reset simulation
        """
        self.tick = 0
        self.sim.reset()
            
    def init_viewer(self,
                    window_width  = None,
                    window_height = None,
                    cam_azimuth   = None,
                    cam_distance  = None,
                    cam_elevation = None,
                    cam_lookat    = None
                    ):
        """
            Initialize viewer
        """
        if not self.VIEWER_EXIST:
            self.VIEWER_EXIST = True
            self.viewer = mujoco_py.MjViewer(self.sim) # this will make a new window
        # Set viewer
        if (window_width is not None) and (window_height is not None):
            self.set_viewer(
                window_width=window_width,window_height=window_height,
                cam_azimuth=cam_azimuth,cam_distance=cam_distance,
                cam_elevation=cam_elevation,cam_lookat=cam_lookat)
        else:
            self.viwer_width = 1000
            self.viwer_height = 600

    def set_viewer(self,
                   window_width  = None,
                   window_height = None,
                   cam_azimuth   = None,
                   cam_distance  = None,
                   cam_elevation = None,
                   cam_lookat    = None
                   ):
        """
            Set viewer
        """
        if self.VIEWER_EXIST:
            if (window_width is not None) and (window_height is not None):
                self.window = self.viewer.window
                self.viwer_width  = int(window_width*get_monitors()[0].width)
                self.viwer_height = int(window_height*get_monitors()[0].height)
                glfw.set_window_size(window=self.window,width=self.viwer_width,height=self.viwer_height)
            # Viewer setting
            if cam_azimuth is not None:
                self.viewer.cam.azimuth = cam_azimuth
            if cam_distance is not None:
                self.viewer.cam.distance = cam_distance
            if cam_elevation is not None:
                self.viewer.cam.elevation = cam_elevation
            if cam_lookat is not None:
                self.viewer.cam.lookat[0] = cam_lookat[0]
                self.viewer.cam.lookat[1] = cam_lookat[1]
                self.viewer.cam.lookat[2] = cam_lookat[2]

    def print_viewer_info(self):
        """
            Print current viewer information
        """
        print ("azimuth:[%.2f] distance:[%.2f] elevation:[%.2f] lookat:[%.2f,%.2f,%.2f]"%(
            self.viewer.cam.azimuth,self.viewer.cam.distance,self.viewer.cam.elevation,
            self.viewer.cam.lookat[0],self.viewer.cam.lookat[1],self.viewer.cam.lookat[2]))

    def get_viewer_info(self):
        """
            Get viewer information
        """
        cam_azimuth   = self.viewer.cam.azimuth
        cam_distance  = self.viewer.cam.distance
        cam_elevation = self.viewer.cam.elevation
        cam_lookat    = self.viewer.cam.lookat
        viewer_info = {
            'cam_azimuth':cam_azimuth,'cam_distance':cam_distance,
            'cam_elevation':cam_elevation,'cam_lookat':cam_lookat
        }
        return viewer_info
            
    def terminate_viewer(self):
        """
            Terminate viewer
        """
        if self.VIEWER_EXIST:
            self.VIEWER_EXIST = False 
            self.viewer.render() # render once before terminate
            time.sleep(1.0)
            glfw.terminate() # terminate
            time.sleep(1.0) 
            glfw.init() # initialize once

    def step(self,ctrl=None,ctrl_idxs=None):
        """
            Step simulation
        """
        # Increase tick
        self.tick = self.tick + 1
        # Control
        if ctrl is not None:
            if ctrl_idxs is None:
                self.sim.data.ctrl[:] = ctrl
            else:
                self.sim.data.ctrl[ctrl_idxs] = ctrl
        # Forward dynamics
        self.sim.step()

    def forward(self,q_pos=None,q_pos_idxs=None,INCREASE_TICK=True):
        """
            Forward kinemaatics
        """
        # Increase tick
        if INCREASE_TICK:
            self.tick = self.tick + 1
        # Forward kinematicaaqs
        if q_pos is not None:
            if q_pos_idxs is None:
                self.sim.data.qpos[:] = q_pos
            else:
                self.sim.data.qpos[q_pos_idxs] = q_pos
        self.sim.forward()
        
    def render(self,RENDER_ALWAYS=False):
        """
            Render simulation
        """
        if RENDER_ALWAYS:
            self.viewer._render_every_frame = True
        else:
            self.viewer._render_every_frame = False
        self.viewer.render()

    def forward_renders(self,max_tick=100):
        """
            Loops of forward and render
        """
        tick = 0
        while tick < max_tick:
            tick = tick + 1
            self.forward(INCREASE_TICK=False)
            self.render()
        
    def get_sim_time(self):
        """
            Get simulation time [sec]
        """
        return self.sim.get_state().time

    def get_q_pos(self,q_pos_idxs=None):
        """
            Get current revolute joint position
        """
        self.sim_state = self.sim.get_state()
        if q_pos_idxs is None:
            q_pos = self.sim_state.qpos[:]
        else:
            q_pos = self.sim_state.qpos[q_pos_idxs]
        return q_pos
        
    def apply_xfrc(self,body_name,xfrc):
        """
            Apply external force (6D) to body
        """
        self.sim.data.xfrc_applied[self.body_name2idx(body_name),:] = xfrc

    def body_name2idx(self,body_name='panda_eef'):
        """
            Body name to index
        """
        return self.sim.model.body_name2id(body_name)

    def body_idx2name(self,body_idx=0):
        """
            Body index to name
        """
        return self.sim.model.body_id2name(body_idx)

    def add_marker(self,pos,type=2,radius=0.02,color=[0.0,1.0,0.0,1.0],label=''):
        """
            Add a maker to renderer
        """
        self.viewer.add_marker(
            pos   = pos,
            type  = type, # mjtGeom: 2:sphere, 3:capsule, 6:box, 9:arrow
            size  = radius*np.ones(3),
            mat   = np.eye(3).flatten(),
            rgba  = color,
            label = label
        )

    def add_arrow(self,pos,uv_arrow,r_stem=0.03,len_arrow=0.3,color=np.array([1,0,0,1]),label=''):
        """
            Add an arrow to renderer
        """
        p_a = np.copy(np.array([0,0,1]))
        p_b = np.copy(uv_arrow)
        p_a_norm = np.linalg.norm(p_a)
        p_b_norm = np.linalg.norm(p_b)
        if p_a_norm > 1e-9: p_a = p_a/p_a_norm
        if p_b_norm > 1e-9: p_b = p_b/p_b_norm
        v = np.cross(p_a,p_b)
        S = np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
        if np.linalg.norm(v) == 0:
            R = np.eye(3,3)
        else:
            R = np.eye(3,3) + S + S@S*(1-np.dot(p_a,p_b))/(np.linalg.norm(v)*np.linalg.norm(v))
        self.viewer.add_marker(pos=pos,size=np.array([r_stem,r_stem,len_arrow]),
                               mat=R,rgba=color,type=mujoco_py.generated.const.GEOM_ARROW,label=label)

    def add_marker_plane(self,p=[0,0,0],R=np.eye(3),xy_widths=[0.5,0.5],rgba=[1.0,0.0,0.0,1.0],label=''):
        """
            Plot plane
        """
        self.viewer.add_marker(
            pos   = p,
            type  = mujoco_py.generated.const.GEOM_PLANE,
            size  = [xy_widths[0],xy_widths[1],0.0],
            mat   = R,
            rgba  = rgba,
            label = label
        )

    def add_marker_sphere(self,p=[0,0,0],radius=0.05,rgba=[1.0,0.0,0.0,1.0],label=''):
        """
            Plot sphere
        """
        self.viewer.add_marker(
            pos   = p,
            type  = mujoco_py.generated.const.GEOM_SPHERE,
            size  = [radius,radius,radius],
            mat   = np.eye(3),
            rgba  = rgba,
            label = label
        )

    def add_marker_box(self,p=[0,0,0],R=np.eye(3),size=[0.1,0.1,0.1],rgba=[1.0,0.0,0.0,1.0],label=''):
        """
            Plot box
        """
        self.viewer.add_marker(
            pos   = p,
            type  = mujoco_py.generated.const.GEOM_BOX,
            size  = size,
            mat   = R,
            rgba  = rgba,
            label = label
        )

    def add_marker_capsule(self,p=[0,0,0],R=np.eye(3),size=[0.1,0.1,0.1],rgba=[1.0,0.0,0.0,1.0],label=''):
        """
            Plot capsule
        """
        self.viewer.add_marker(
            pos   = p,
            type  = mujoco_py.generated.const.GEOM_CAPSULE,
            size  = size,
            mat   = R,
            rgba  = rgba,
            label = 'Capsule'
        )
    
    def add_marker_cylinder(self,p=[0,0,0],R=np.eye(3),size=[0.1,0.1,0.1],rgba=[1.0,0.0,0.0,1.0],label=''):
        """
            Plot cylinder
        """
        self.viewer.add_marker(
            pos   = p,
            type  = mujoco_py.generated.const.GEOM_CYLINDER,
            size  = size,
            mat   = R,
            rgba  = rgba,
            label = label
        )

    def add_marker_arrow(self,p=[0,0,0],R=np.eye(3),size=[0.1,0.1,0.1],rgba=[1.0,0.0,0.0,1.0],label=''):
        """
            Plot arrow
        """
        self.viewer.add_marker(
            pos   = p,
            type  = mujoco_py.generated.const.GEOM_ARROW,
            size  = size,
            mat   = R,
            rgba  = rgba,
            label = 'Arrow'
        )

    def add_marker_coordinate(self,p=[0,0,0],R=np.eye(3),axis_len=0.5,axis_width=0.01,rgba=None,label=''):
        """
            Plot coordinate
        """
        if rgba is None:
            rgba_x = [1.0,0.0,0.0,0.9]
            rgba_y = [0.0,1.0,0.0,0.9]
            rgba_z = [0.0,0.0,1.0,0.9]
        else:
            rgba_x = rgba
            rgba_y = rgba
            rgba_z = rgba
        R_x = R@rpy2r(np.deg2rad([0,0,90]))@rpy2r(np.pi/2*np.array([1,0,0]))
        p_x = p + R_x[:,2]*axis_len/2
        self.viewer.add_marker(
            pos   = p_x,
            type  = mujoco_py.generated.const.GEOM_CYLINDER,
            size  = [axis_width,axis_width,axis_len/2],
            mat   = R_x,
            rgba  = rgba_x,
            label = ''
        )
        R_y = R@rpy2r(np.deg2rad([0,0,90]))@rpy2r(np.pi/2*np.array([0,1,0]))
        p_y = p + R_y[:,2]*axis_len/2
        self.viewer.add_marker(
            pos   = p_y,
            type  = mujoco_py.generated.const.GEOM_CYLINDER,
            size  = [axis_width,axis_width,axis_len/2],
            mat   = R_y,
            rgba  = rgba_y,
            label = ''
        )
        R_z = R@rpy2r(np.deg2rad([0,0,90]))@rpy2r(np.pi/2*np.array([0,0,1]))
        p_z = p + R_z[:,2]*axis_len/2
        self.viewer.add_marker(
            pos   = p_z,
            type  = mujoco_py.generated.const.GEOM_CYLINDER,
            size  = [axis_width,axis_width,axis_len/2],
            mat   = R_z,
            rgba  = rgba_z,
            label = ''
        )
        self.add_marker_sphere(p=p,radius=0.001,rgba=[1.0,1.0,1.0,1.0],label=label)

    def get_p_body(self,body_name):
        """
            Get body position
        """
        self.sim_state = self.sim.get_state()
        p = np.array(self.sim.data.body_xpos[self.body_name2idx(body_name)])
        return p

    def get_R_body(self,body_name):
        """
            Get body rotation
        """
        self.sim_state = self.sim.get_state()
        R = np.array(self.sim.data.body_xmat[self.body_name2idx(body_name)].reshape([3, 3]))
        return R

    def get_J_body(self,body_name):
        """
            Get body Jacobian
        """
        J_p    = np.array(self.sim.data.get_body_jacp(body_name).reshape((3, -1))[:,self.rev_qvel_idxs])
        J_R    = np.array(self.sim.data.get_body_jacr(body_name).reshape((3, -1))[:,self.rev_qvel_idxs])
        J_full = np.array(np.vstack([J_p,J_R]))
        return J_p,J_R,J_full

    def one_step_ik(self,body_name,p_trgt=None,R_trgt=None,stepsize=5.0*np.pi/180.0,eps=1e-6):
        """
            One-step inverse kinematics
        """
        J_p,J_R,J_full = self.get_J_body(body_name=body_name)
        p_curr = self.get_p_body(body_name=body_name)
        R_curr = self.get_R_body(body_name=body_name)
        if (p_trgt is not None) and (R_trgt is not None): # both p and R targets are given
            p_err = (p_trgt-p_curr)
            R_err = np.linalg.solve(R_curr,R_trgt)
            w_err = R_curr @ r2w(R_err)
            J,err = J_full,np.concatenate((p_err,w_err))
        elif (p_trgt is not None) and (R_trgt is None): # only p target is given
            p_err = (p_trgt-p_curr)
            J,err = J_p,p_err
        elif (p_trgt is None) and (R_trgt is not None): # only R target is given
            R_err = np.linalg.solve(R_curr,R_trgt)
            w_err = R_curr @ r2w(R_err)
            J,err = J_R,w_err
        else:
            raise Exception('At least one IK target is required!')
        # Compute dq using least-square
        dq = np.linalg.solve(a=(J.T@J)+eps*np.eye(J.shape[1]),b=J.T@err)
        # Trim scale 
        dq = trim_scale(x=dq,th=stepsize)
        return dq,err

    def backup_sim_data(self,joint_idxs=None):
        """
            Backup sim data (qpos, qvel, qacc)
        """
        if joint_idxs is None:
            joint_idxs = self.rev_joint_idxs
        self.qpos_bu = self.sim.data.qpos[joint_idxs]
        self.qvel_bu = self.sim.data.qvel[joint_idxs]
        self.qacc_bu = self.sim.data.qacc[joint_idxs]

    def restore_sim_data(self,joint_idxs=None):
        """
            Restore sim data (qpos, qvel, qacc)
        """
        if joint_idxs is None:
            joint_idxs = self.rev_joint_idxs
        self.sim.data.qpos[joint_idxs] = self.qpos_bu
        self.sim.data.qvel[joint_idxs] = self.qvel_bu
        self.sim.data.qacc[joint_idxs] = self.qacc_bu

    def solve_inverse_dynamics(self,qvel,qacc,joint_idxs=None):
        """
            Solve inverse dynamics to get torque from qvel and qacc
        """
        if joint_idxs is None:
            joint_idxs = self.rev_joint_idxs
        # Backup
        self.backup_sim_data(joint_idxs=joint_idxs)
        # Compute torque
        self.sim.data.qpos[joint_idxs] = self.get_q_pos(q_pos_idxs=joint_idxs)
        self.sim.data.qvel[joint_idxs] = qvel
        self.sim.data.qacc[joint_idxs] = qacc
        mujoco_py.functions.mj_inverse(self.sim.model,self.sim.data)
        torque = self.sim.data.qfrc_inverse[joint_idxs].copy()
        # Restore
        self.restore_sim_data(joint_idxs=joint_idxs)
        return torque
    
    def get_contact_infos(self):
        """
            Get contact information
        """
        n_contact = self.sim.data.ncon
        contact_infos = []
        for c_idx in range(n_contact):
            contact = self.sim.data.contact[c_idx]
            # Compute contact point and force
            p_contact = contact.pos
            f_contact = np.zeros(6,dtype=np.float64) 
            mujoco_py.functions.mj_contactForce(self.sim.model,self.sim.data,c_idx,f_contact)
            # The contact force is in the contact frame
            contact_frame = contact.frame
            R_frame = contact_frame.reshape((3,3))
            f_contact_global = R_frame @ f_contact[:3]
            f_norm = np.linalg.norm(f_contact_global)
            # Contacting bodies
            bodyid1 = self.sim.model.geom_bodyid[contact.geom1]
            bodyid2 = self.sim.model.geom_bodyid[contact.geom2]
            bodyname1 = self.body_idx2name(bodyid1)
            bodyname2 = self.body_idx2name(bodyid2)
            # Append
            contact_infos.append(
                {'p':p_contact,'f':f_contact_global,'f_norm':f_norm,
                 'bodyname1':bodyname1,'bodyname2':bodyname2}
                )
        return contact_infos

def get_env_obj_names(env,prefix='obj_'):
    """
        Accumulate object names by assuming that the prefix is 'obj_'
    """
    obj_names = [x for x in env.joint_names if x[:len(prefix)]==prefix]
    return obj_names

def set_env_obj(
    env,
    obj_name  = 'obj_box_01',
    obj_pos   = [1.0,0.0,0.75],
    obj_quat  = [0,0,0,1],
    obj_color = None
    ):
    """
        Set a single object in an environment
    """
    # Get address
    qpos_addr = env.sim.model.get_joint_qpos_addr(obj_name)
    # Set position
    env.sim.data.qpos[qpos_addr[0]]   = obj_pos[0] # x
    env.sim.data.qpos[qpos_addr[0]+1] = obj_pos[1] # y
    env.sim.data.qpos[qpos_addr[0]+2] = obj_pos[2] # z
    # Set rotation
    env.sim.data.qpos[qpos_addr[0]+3:qpos_addr[1]] = obj_quat # quaternion
    # Color
    if obj_color is not None:
        idx = env.sim.model.geom_name2id(obj_name)
        env.sim.model.geom_rgba[idx,:] = obj_color

def set_env_objs(
    env,
    obj_names,
    obj_poses,
    obj_colors=None):
    """
        Set multiple objects
    """
    for o_idx,obj_name in enumerate(obj_names):
        obj_pos = obj_poses[o_idx,:]
        if obj_colors is not None:
            obj_color = obj_colors[o_idx,:]
        else:
            obj_color = None
        set_env_obj(env,obj_name=obj_name,obj_pos=obj_pos,obj_color=obj_color)


def get_env_obj_poses(env,obj_names):
    """
        Get object poses 
    """
    n_obj     = len(obj_names)
    obj_ps = np.zeros(shape=(n_obj,3))
    obj_Rs = np.zeros(shape=(n_obj,3,3))
    for o_idx,obj_name in enumerate(obj_names):
        qpos_addr = env.sim.model.get_joint_qpos_addr(obj_name)
        # Get position
        x = env.sim.data.qpos[qpos_addr[0]]
        y = env.sim.data.qpos[qpos_addr[0]+1]
        z = env.sim.data.qpos[qpos_addr[0]+2]
        # Set rotation (upstraight)
        quat = env.sim.data.qpos[qpos_addr[0]+3:qpos_addr[1]]
        R = quat2r(quat)
        # Append
        obj_ps[o_idx,:] = np.array([x,y,z])
        obj_Rs[o_idx,:,:] = R
    return obj_ps,obj_Rs

## Get Robot Body names.
def get_env_body_names(env,prefix='ur_'):
    """
        Accumulate robot body names by assuming that the prefix is 'ur_'
    """
    body_names = [x for x in env.body_names if x[:len(prefix)]==prefix]

    return body_names
    
def get_base2ee_matrix(env, link_prefix='ur_', verbose=False):
    """
        In AX=XB Equation, (extrinsic calibration) 
        Get matrix about B that represents sequenced transformation operations on [Robot base to Robot End-Effector].
    """
    # ur_links = ['ur_base_link', 'ur_shoulder_link', 'ur_upper_arm_link', 'ur_forearm_link', 'ur_wrist_1_link', 'ur_wrist_2_link', 'ur_wrist_3_link']

    link_names = get_env_body_names(env, link_prefix)
    T_links = []

    for idx, link in enumerate(link_names):
        if verbose == True:
            print(link)
        p_link = env.get_p_body(body_name=link)  # 3x3
        R_link = env.get_R_body(body_name=link)  # 3x1

        # T_link = pr2t(p_link, R_link)
        T_link = cv2.hconcat((R_link, p_link))      # 3x4
        T_link = np.vstack((T_link, np.array([0,0,0,1])))   # 4x4
        T_links.append(T_link)

    # start at 'ur_base_link'
    T_bs2end = T_links[0]
    for i in range(len(T_links)-1):
        T_bs2end = np.matmul(T_bs2end, T_links[i+1])

    return T_bs2end

# Get apriltag pose
def get_apriltag_pose(env, img, img_depth):
    """
        In AX=XB Equation, (extrinsic calibration) 
        Get matrix about A that represents detected AprilTag pose in camera coordinate.
    """
    detector = apriltag.Detector()
    img_BGR = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_Gray = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY)

    results = detector.detect(img_Gray)

    cam_matrix, _, _ = env.camera_matrix_and_pose(width=env.render_width, height=env.render_height, camera_name="main1")

    fx = cam_matrix[0][0]
    cx = cam_matrix[0][2]
    fy = cam_matrix[1][1]
    cy = cam_matrix[1][2]

    cam_params = [fx, fy, cx, cy]

    img_real = np.array(env.depth_2_meters(img_depth))
    img_xyz = compute_xyz(img_real, cam_matrix=cam_matrix)

    # Render the detections on the image
    if len(results) > 0:
        draw_bbox(results, img, verbose=False)

        for r in results:
            pose, e0, e1 = detector.detection_pose(detection=r, camera_params=cam_params, tag_size=0.06)    # should check tag_size
            
            poseRotation = pose[:3, :3]
            poseTranslation = pose[:3, 3]
    
            center_point = [int(r.center[i]) for i in range(2)]    # in int type

            rot_april = pose[:3, :3]
            center_3d = np.array([img_xyz[center_point[1]][center_point[0]]])   # order of pixel array is y, x 

            T_april = np.concatenate((rot_april, center_3d.T), axis=1)  # 4x3 matrix
            T_april = np.concatenate((T_april, np.array([[0,0,0,1]])), axis=0)  # 4x4 matrix

        return T_april

    else:   # if any detected marker is none, return None.
        return None


## Get/Set Robot Joint variables
def get_env_joint_names(env,prefix='ur_'):
    """
        Accumulate robot joint names by assuming that the prefix is 'ur_'
    """
    joint_names = [x for x in env.joint_names if x[:len(prefix)]==prefix]

    return joint_names

def print_env_joint_infos(env, prefix='ur_'):
    joint_names = get_env_obj_names(env, prefix) # available objects
    for joint_idx,joint_name in enumerate(joint_names):
        qpos_addr = env.sim.model.get_joint_qpos_addr(joint_name)
        print ("[%d/%d] joint_name:[%s]"%(joint_idx,len(joint_names),joint_name))
        print ("[%d/%d] joint_configurations:[%0.3f]"%(joint_idx,len(joint_names),env.sim.data.qpos[qpos_addr]))    

def set_env_joint_configuration(env, configurations, prefix='ur_'):
    """
        Set robot joint poses 
    """
    joint_names = get_env_joint_names(env, prefix=prefix)
    assert len(configurations) == len(joint_names)

    for joint_idx,joint_name in enumerate(joint_names):
        # Get address
        qpos_addr = env.sim.model.get_joint_qpos_addr(joint_name)

        # Set configurations
        env.sim.data.qpos[qpos_addr] = configurations[joint_idx]


def random_spawn_objects(
    env,
    prefix     = 'obj_',
    x_init     = -1.0,
    n_place    = 5,
    x_range    = [0.5,1.5],
    y_range    = [-0.4,0.4],
    z_range    = [0.8,0.85],
    min_dist   = 0.2,
    ):
    """
        Randomly spawn objects
    """
    # Reset
    env.reset() 
    # Place objects in a row on the ground
    obj_names = get_env_obj_names(env,prefix=prefix) # available objects
    # colors = [plt.cm.gist_rainbow(x) for x in np.linspace(0,1,len(obj_names))]
    for obj_idx,obj_name in enumerate(obj_names):
        obj_pos   = [x_init,0.1*obj_idx,0.0]
        obj_quat  = [0,0,0,1]
        # obj_color = colors[obj_idx]
        set_env_obj(env=env,obj_name=obj_name,obj_pos=obj_pos,obj_quat=obj_quat,obj_color=None)
    env.forward(INCREASE_TICK=False) # update object locations

    # Randomly place objects on the table
    obj2place_idxs = np.random.permutation(len(obj_names))[:n_place].astype(int)
    obj2place_names = [obj_names[o_idx] for o_idx in obj2place_idxs]
    obj2place_poses = np.zeros((n_place,3))
    for o_idx in range(n_place):
        while True:
            x = np.random.uniform(low=x_range[0],high=x_range[1])
            y = np.random.uniform(low=y_range[0],high=y_range[1])
            z = np.random.uniform(low=z_range[0],high=z_range[1])
            xyz = np.array([x,y,z])
            if o_idx >= 1:
                devc = cdist(xyz.reshape((-1,3)),obj2place_poses[:o_idx,:].reshape((-1,3)),'euclidean')
                if devc.min() > min_dist: break # minimum distance between objects
            else:
                break
        obj2place_poses[o_idx,:] = xyz
    set_env_objs(env,obj_names=obj2place_names,obj_poses=obj2place_poses,obj_colors=None)
    env.forward()

def get_viewer_coordinate(cam_lookat,cam_distance,cam_elevation,cam_azimuth):
    """
        Get viewer coordinate 
    """
    p_lookat = cam_lookat
    R_lookat = rpy2r(np.deg2rad([0,-cam_elevation,cam_azimuth]))
    T_lookat = pr2t(p_lookat,R_lookat)
    T_viewer = T_lookat @ pr2t(np.array([-cam_distance,0,0]),np.eye(3)) # minus translate w.r.t. x
    return T_viewer,T_lookat

def depth2pcd(depth):
    # depth = remap(depth, depth.min(), depth.max(), 0, 1)    # re mapping of depth
    # print(depth)
    scalingFactor = 1
    fovy          = 45 # default value is 45.
    aspect        = depth.shape[1] / depth.shape[0]
    fovx          = 2 * math.atan(math.tan(fovy * 0.5 * math.pi / 360) * aspect)
    width         = depth.shape[1]
    height        = depth.shape[0]
    fovx          = 2 * math.atan(width * 0.5 / (height * 0.5 / math.tan(fovy * math.pi / 360 / 2))) / math.pi * 360
    fx            = width / 2 / (math.tan(fovx * math.pi / 360))
    fy            = height / 2 / (math.tan(fovy * math.pi / 360))
    points = []
    for v in range(0, height, 10):
        for u in range(0, width, 10):
            Z = depth[v][u] / scalingFactor
            if Z == 0:
                continue
            X = (u - width / 2) * Z / fx
            Y = (v - height / 2) * Z / fy
            points.append([X, Y, Z])
    return np.array(points)


from matplotlib import animation
from IPython.display import display, HTML

def display_animation(anim):
    plt.close(anim._fig)
    return HTML(anim.to_jshtml())

def display_frames_as_gif(frame_list, filename):
    patch = plt.imshow(frame_list[0])
    plt.axis('off')
    def animate(i):
        patch.set_data(frame_list[i])
    anim = animation.FuncAnimation(
        plt.gcf(),animate,frames=len(frame_list),interval=20)
    display(display_animation(anim))
    # plt.gcf().savefig("gifs.gif")
    animation.Animation.save(anim, filename, fps=30)
print ("Done.")


def draw_bbox(results, image, verbose=False):
    width = 1500
    height = 1000

    for r in results:
        # extract the bounding box (x, y)-coordinates for the AprilTag
        # and convert each of the (x, y)-coordinate pairs to integers
        (ptA, ptB, ptC, ptD) = r.corners
        ptB = (int(ptB[0]), int(ptB[1]))
        ptC = (int(ptC[0]), int(ptC[1]))
        ptD = (int(ptD[0]), int(ptD[1]))
        ptA = (int(ptA[0]), int(ptA[1]))

        # draw the bounding box of the AprilTag detection
        cv2.line(image, ptA, ptB, (255, 0, 0), 3)
        cv2.line(image, ptB, ptC, (255, 0, 0), 3)
        cv2.line(image, ptC, ptD, (255, 0, 0), 3)
        cv2.line(image, ptD, ptA, (255, 0, 0), 3)

        # draw the center (x, y)-coordinates of the AprilTag
        (cX, cY) = (int(r.center[0]), int(r.center[1]))
        cv2.circle(image, (cX, cY), 5, (0, 0, 255), -1)

        # draw the tag family on the image
        tagFamily = r.tag_family.decode("utf-8")
    
        if verbose == True:
            cv2.putText(image, tagFamily, (ptA[0], ptA[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 5.0, (255, 255, 255), 3)
            # print("Apriltag name: {}".format(tagFamily))

            x_centered = cX - width / 2
            y_centered = -1 * (cY - height / 2)

            cv2.putText(image, f"Center X coord: {x_centered}", (ptB[0] + 10, ptB[1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (125, 0, 125), 7)

            cv2.putText(image, f"Center Y coord: {y_centered}", (ptB[0] + 10, ptB[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (125, 0, 125), 7)

            cv2.putText(image, f"Tag ID: {r.tag_id}", (ptC[0] - 70, ptC[1] - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (125, 0, 125), 7)

        cv2.circle(image, (int((width / 2)), int((height / 2))), 5, (0, 0, 255), 2)


def convert_from_uvd(u, v, d, cam_matrix):
    """
        pxToMetre: Constant, depth scale factor
        cx: Center x of Camera
        cy: Center y of Camera
        focalx: Focal length
        focaly: Focal length 
    """

    pxToMetre = 1

    focalx = cam_matrix[0][0]
    cx = cam_matrix[0][2]
    focaly = cam_matrix[1][1]
    cy = cam_matrix[1][2]

    d *= pxToMetre
    x_over_z = (cx - u) / focalx
    y_over_z = (cy - v) / focaly
    z = d / np.sqrt(1. + x_over_z**2 + y_over_z**2)
    x = x_over_z * z
    y = y_over_z * z

    return -y, x, z

# def compute_xyz(depth_img, fx, fy, px, py, height, width):
#     indices = np.indices((height, width), dtype=np.float32).transpose(1,2,0)
#     z_e = depth_img
#     x_e = (indices[..., 1] - px) * z_e / fx
#     y_e = (indices[..., 0] - py) * z_e / fy
    
#     # Order of y_ e is reversed !
#     xyz_img = np.stack([-y_e, x_e, z_e], axis=-1) # Shape: [H x W x 3]
#     return xyz_img

def compute_xyz(depth_img, cam_matrix):

    # , fx, fy, px, py, height, width
    fx = cam_matrix[0][0]
    cx = cam_matrix[0][2]
    fy = cam_matrix[1][1]
    cy = cam_matrix[1][2]

    height = 1000
    width = 1500

    indices = np.indices((height, width), dtype=np.float32).transpose(1, 2, 0)
    
    z_e = depth_img
    x_e = (indices[..., 1] - cx) * z_e / fx
    y_e = (indices[..., 0] - cy) * z_e / fy
    
    # Order of y_ e is reversed !
    xyz_img = np.stack([-y_e, x_e, z_e], axis=-1) # Shape: [H x W x 3]
    return xyz_img

## Need to check
def d2_to_d3(x = 0, y = 0):
    """
    Conversion about 2D image to 3D point
    """

    rot = 0

    ## x and y varies from -1 to 1
    ## return XYZ in the frame attach to camera link allined with base link

    fx = 1/math.tan(math.radians(45.0/2.0))
    theta_x = math.radians(rot) + math.atan2(x, fx)
    X = 1.5 * math.tan(theta_x)
    Z = 1.5
    fy = 1/math.tan(math.radians(45.0/2.0))
    theta_y = math.atan2(y,fy)
    Y = math.tan(theta_y) * 1.5 / math.cos(theta_x)
    
    return X,Y,Z

def rotation_matrix_to_spherical_rotation(R):
    # Extract rotation about the z-axis
    azimuth = np.arctan2(R[1,0], R[0,0])

    # Extract rotation about the y-axis
    elevation = np.arctan2(-R[2,0], np.sqrt(R[2,1]**2 + R[2,2]**2))

    # Extract rotation about the x-axis
    roll = np.arctan2(R[2,1], R[2,2])
    
    return azimuth, elevation
