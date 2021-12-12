#!/usr/bin/python

import numpy as np

# Input: expects 3xN matrix of points
# Returns R,t
# R = 3x3 rotation matrix
# t = 3x1 column vector

def rigid_transform_3D(A, B):
    assert A.shape == B.shape

    A = A.T
    B= B.T   

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    #if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    M = np.empty((4, 4))
    M[:3, :3] = R
    M[:3, 3] = t.reshape(-1)
    M[3, :] = [0, 0, 0, 1]

    return M

p1_t = np.array([[0,0,0], [1,0,0],[0,1,0],[0,1,0]])

p1 = np.concatenate([p1_t, np.ones((p1_t.shape[0],1))],1)
print(p1)
p2_t = np.array([[0,0,1], [1,0,1],[0,0,2],[0,0,2]]) #Approx transformation is 90 degree rot over x-axis and +1 in Z axis

M = rigid_transform_3D(p1_t,p2_t)
B2 = (M@p1.T).T[:,:3]
print(B2-p2_t)