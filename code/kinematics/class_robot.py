import numpy as np 
import rospy 
from kinematics.utils.util_ik import make_ik_input
from kinematics.class_structure import CHAIN

class RobotClass(object):
    def __init__(self, file_name, base_offset):
        super(RobotClass,self).__init__() 
        self.chain = CHAIN(file_name=file_name, base_offset=base_offset)
        self.chain.add_joi_to_robot()
        self.chain.add_link_to_robot()
        self.chain.fk_chain(1)
    
    def solve_ik(self,  target_name      = ['wrist_3_joint'],
                        target_position  = [[0,0,0]],
                        target_rotation  = [[-1.57, 0, -1.57]],
                        solve_position   = [1],
                        solve_rotation   = [1],
                        weight_position  = 1,
                        weight_rotation  = 1,
                        joi_ctrl_num= 6): 
                        
        ingredients = make_ik_input(target_name = target_name,
                                    target_pos  = target_position,
                                    target_rot  = target_rotation,
                                    solve_pos   = solve_position,
                                    solve_rot   = solve_rotation,
                                    weight_pos  = weight_position,
                                    weight_rot  = weight_rotation,
                                    joi_ctrl_num= joi_ctrl_num)
        total_q = self.chain.get_q_from_ik(ingredients)
        ctrl_q  = total_q[:joi_ctrl_num+1] #Including base joint for Unity Env
        ctrl_q  = ctrl_q.reshape(-1,)
        return ctrl_q 