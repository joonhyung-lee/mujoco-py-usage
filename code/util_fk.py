import numpy as np 
import math
import scipy

def rot_e():
    e = np.array([[1, 	       0, 	      0],
             	  [0,          1,         0],
             	  [0,          0,         1]])
    return e


def rot_x(rad):
    roll = np.array([[1, 	       0, 	         0],
             		 [0, np.cos(rad), -np.sin(rad)],
             		 [0, np.sin(rad),  np.cos(rad)]])
    return roll 


def rot_y(rad):
    pitch = np.array([[np.cos(rad), 0, np.sin(rad)],
                      [0,		    1, 	         0],
                      [-np.sin(rad),0, np.cos(rad)]])
    return pitch


def rot_z(rad):
    yaw = np.array([[np.cos(rad), -np.sin(rad),  0],
         	        [np.sin(rad),  np.cos(rad),  0],
              		[0, 			         0,  1]])
    return yaw 


def Rotation_E(): 
    e = np.array([[1, 	       0, 	      0,    0],
             	  [0,          1,         0,    0],
             	  [0,          0,         1,    0],
             	  [0,		   0,	      0,    0]])
    return e


def Rotation_X(rad):
    roll = np.array([[1, 	       0, 	      0,    0],
             		 [0, np.cos(rad), -np.sin(rad), 0],
             		 [0, np.sin(rad),  np.cos(rad), 0],
             		 [0,		   0,	      0,    0]])
    return roll 


def Rotation_Y(rad):
    pitch = np.array([[np.cos(rad), 0, np.sin(rad), 0],
              		  [0,		    1, 	         0, 0],
              		  [-np.sin(rad),0, np.cos(rad), 0],
              		  [0, 		    0, 	         0, 0]])
    return pitch


def Rotation_Z(rad):
    yaw = np.array([[np.cos(rad), -np.sin(rad),  0, 0],
         	        [np.sin(rad),  np.cos(rad),  0, 0],
              		[0, 			         0,  1, 0],
             		[0, 			         0,  0, 0]])
    return yaw 

def Translation(x , y, z):
    Position = np.array([[0, 0, 0, x],
                         [0, 0, 0, y],
                         [0, 0, 0, z],
                         [0, 0, 0, 1]])
    return Position


def HT_matrix(Rotation, Position):
    Homogeneous_Transform = Rotation + Position
    return Homogeneous_Transform


def pr2t(position, rotation): 
    position_4diag  = np.array([[0, 0, 0, position[0]],
                                [0, 0, 0, position[1]],
                                [0, 0, 0, position[2]], 
                                [0, 0, 0, 1]], dtype=object)
    rotation_4diag  = np.append(rotation,[[0],[0],[0]], axis=1)
    rotation_4diag_ = np.append(rotation_4diag, [[0, 0, 0, 1]], axis=0)
    ht_matrix = position_4diag + rotation_4diag_ 
    return ht_matrix


def t2p(ht_matrix):
    return ht_matrix[:-1, -1]


def t2r(ht_matrix):
    return ht_matrix[:-1, :-1]


def make_rotation(rad=0):
    for idx, rad_num in enumerate(rad.split()):
        if idx == 0 and float(rad_num) !=0:
            idx0 = rot_x(float(rad_num))
        elif idx==0 and float(rad_num) == 0: 
            idx0 = rot_e()
        if idx == 1 and float(rad_num) !=0:
            idx1 = rot_y(float(rad_num))
        elif idx==1 and float(rad_num) == 0: 
            idx1 = rot_e()
        if idx == 2 and float(rad_num) !=0:
            idx2 = rot_z(float(rad_num))
        elif idx==2 and float(rad_num)==0: 
            idx2 = rot_e()
    rot = idx2.dot(idx1).dot(idx0) 
    return rot

def camera_to_base(transform_mat, points):
    # points: (x,y,z)
    ones = np.ones((len(points),1))
    points = np.concatenate((points,ones),axis=1)
    t_points = points.T     # point: 
    t_transformed_ponints = np.dot(transform_mat,t_points)
    transformed_ponints = t_transformed_ponints.T
    xyz = transformed_ponints[:,0:3]
    return xyz


def log_group_skew(T):
    """
        Convert to Log matrix form. Return skew matrix form.
        This is code of Lemma 2.
    """
    R = T[0:3, 0:3] # Rotation matrix
    # Lemma 2
    theta = np.arccos((np.trace(R) - 1)/2)

    logr = np.array([R[2,1] - R[1,2], R[0,2] - R[2,0], R[1,0] - R[0,1]]) * theta / (2*np.sin(theta))
    return logr # return skew matrix form

def log_group_matrix(T):
    """
        Convert to Log matrix form. Return 3x3 matrix form.
        This is code of Lemma 2.
    """
    R = T[0:3, 0:3] # Rotation matrix
    # Lemma 2
    theta = np.arccos((np.trace(R) - 1)/2)

    theta_ = R - R.T
    log_r = theta_ * theta / (2*np.sin(theta))
    return log_r

def calibrate(A, B):
    """
        Solve AX=XB calibration.
    """

    data_num = len(A)
    
    M = np.zeros((3,3))
    C = np.zeros((3*data_num, 3))
    d = np.zeros((3*data_num, 1))

    # columns of matrix A, B
    alpha = log_group_skew(A[0])     # 3 x 1
    beta = log_group_skew(B[0])      # 3 x 1
    alpha_2 = log_group_skew(A[1]) # 3 x 1
    beta_2 = log_group_skew(B[1])  # 3 x 1
    alpha_3 = np.cross(alpha, alpha_2)  # 3 x 1
    beta_3 = np.cross(beta, beta_2)     # 3 x 1

    # print(alpha)
    # print(alpha_2)
    # print(beta)
    # print(beta_2)
    # assert np.array_equal(np.cross(alpha, alpha_2), np.zeros(3)) 
    # assert np.array_equal(np.cross(beta, beta_2), np.zeros(3))

    # M = \Sigma (beta * alpha.T)
    M1 = np.dot(beta.reshape(3,1),alpha.reshape(3,1).T)
    M2 = np.dot(beta_2.reshape(3,1),alpha_2.reshape(3,1).T)
    M3 = np.dot(beta_3.reshape(3,1),alpha_3.reshape(3,1).T)
    M = M1+M2+M3

    # theta_x = (M.T * M)^(-1/2) * M.T    
    # theta_x = np.dot(np.sqrt(np.linalg.inv((np.dot(M.T, M)))), M.T)
    # RuntimeWarning: invalid value encountered in sqrt: np.sqrt results nan values
    theta_x = np.dot(scipy.linalg.sqrtm(np.linalg.inv((np.dot(M.T, M)))), M.T)  # rotational info

    # A_ = np.array([alpha, alpha_2, alpha_3])
    # B_ = np.array([beta, beta_2, beta_3])
    # B_inv = np.linalg.inv(B_)
    # theta_x = A * B_inv

    for i in range(data_num):
        A_rot   = A[i][0:3,0:3]
        A_trans = A[i][0:3, 3]
        B_rot   = B[i][0:3,0:3]
        B_trans = B[i][0:3, 3]
        
        C[3*i:3*i+3, :] = np.eye(3) - A_rot
        d[3*i:3*i+3, 0] = A_trans - np.dot(theta_x, B_trans)


    b_x = np.dot(np.linalg.inv(np.dot(C.T, C)), np.dot(C.T, d))     # translational info

    return theta_x, b_x


def T2axisangle(T):
    """
        T to axis-angle representation.
    """   
    R = T[:3,:3]

    theta_axisangle = math.acos((np.trace(R)-1)/2)    # in rad unit.

    prefix_multiplier = 1 / (2*math.sin(theta_axisangle))

    rx = prefix_multiplier * (R[2][1] - R[1][2]) * theta_axisangle
    ry = prefix_multiplier * (R[0][2] - R[2][0]) * theta_axisangle
    rz = prefix_multiplier * (R[1][0] - R[0][1]) * theta_axisangle

    rot_axisangle = np.array([rx, ry, rz])

    return rot_axisangle, theta_axisangle

def skew(R):    # return ske
    """
        Convert to skew matrix.
    """   
    return np.array([[0, -R[2], R[1]],
                     [R[2], 0, -R[0]],
                     [-R[1], R[0], 0]])


def T2aa(T):
    T = np.array(T, dtype=np.float64, copy=False)

    rot = T[:3, :3]
    # direction: unit eigenvector of R33 corresponding to eigenvalue of 1
    w, W = np.linalg.eig(rot.T)
    
    i = np.where(abs(np.real(w) - 1.0) < 1e-8)[0]
    if not len(i):
        raise ValueError("no unit eigenvector corresponding to eigenvalue 1")
    
    axis = np.real(W[:, i[-1]]).squeeze()
    # point: unit eigenvector of R corresponding to eigenvalue of 1
    w, Q = np.linalg.eig(T)
    
    i = np.where(abs(np.real(w) - 1.0) < 1e-8)[0]
    if not len(i):
        raise ValueError("no unit eigenvector corresponding to eigenvalue 1")

    point = np.real(Q[:, i[-1]]).squeeze()
    point /= point[3]

    # rotation angle depending on axis
    cosa = (np.trace(rot) - 1.0) / 2.0
    if abs(axis[2]) > 1e-8:
        sina = (T[1, 0] + (cosa-1.0)*axis[0]*axis[1]) / axis[2]
    elif abs(axis[1]) > 1e-8:
        sina = (T[0, 2] + (cosa-1.0)*axis[0]*axis[2]) / axis[1]
    else:
        sina = (T[2, 1] + (cosa-1.0)*axis[1]*axis[2]) / axis[0]

    angle = math.atan2(sina, cosa)
    return axis, angle, point
