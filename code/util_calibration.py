import numpy as np
import scipy
from util_fk import r2axisangle, skew


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

def get_extrinsic_calibration_park(A, B):
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


def estimate_translation(A, B, Rx):
    """
    Estimate the translation component of :math:`\hat{X}` in :math:`AX=XB`. This
    requires the estimation of the rotation component :math:`\hat{R}_x`
    Parameters
    ----------
    A: list
        List of homogeneous transformations with the relative motion of the
        end-effector
    B: list
        List of homogeneous transformations with the relative motion of the
        calibration pattern (often called `object`)
    Rx: array_like
        Estimate of the rotation component (rotation matrix) of :math:`\hat{X}`
    Returns
    -------
    tx: array_like
        The estimated translation component (XYZ value) of :math:`\hat{X}`
    """
    C = []
    d = []
    for Ai, Bi in zip(A, B):
        ta = Ai[:3, 3]
        tb = Bi[:3, 3]
        C.append(Ai[:3, :3]-np.eye(3))
        d.append(np.dot(Rx, tb)-ta)
    C = np.array(C)
    C.shape = (-1, 3)
    d = np.array(d).flatten()
    tx, residuals, rank, s = np.linalg.lstsq(C, d, rcond=-1)
    
    return tx.flatten()


def get_extrinsic_calibration_tsai(A, B):
    """
        Implementation of Tsai method Extrinsic calibration.
    """
    norm = np.linalg.norm
    C = []
    d = []

    for Ai, Bi in zip(A, B):
        # Transform the matrices to their axis-angle representation
        r_gij, theta_gij = r2axisangle(Ai)
        r_cij, theta_cij = r2axisangle(Bi)

        # Tsai uses a modified version of the angle-axis representation
        Pgij = 2*np.sin(theta_gij/2.)*r_gij
        Pcij = 2*np.sin(theta_cij/2.)*r_cij

        # Use C and d to avoid overlapping with the input A-B
        C.append(skew(Pgij+Pcij))
        d.append(Pcij-Pgij)

    # Estimate Rx
    C = np.array(C)
    C.shape = (-1, 3)

    d = np.array(d).flatten()

    Pcg_, residuals, rank, s = np.linalg.lstsq(C, d, rcond=-1)
    Pcg = 2*Pcg_ / np.sqrt(1 + norm(Pcg_)**2)

    R1 = (1 - norm(Pcg)**2/2.) * np.eye(3)
    R2 = (np.dot(Pcg.reshape(3, 1), Pcg.reshape(1, 3)) +
            np.sqrt(4-norm(Pcg)**2) * skew(Pcg)) / 2.
    Rx = R1 + R2

    # Estimate tx
    tx = estimate_translation(A, B, Rx)
    
    # Return X
    X = np.eye(4)
    X[:3, :3] = Rx
    X[:3, 3] = tx
    return X