import numpy as np
import cv2
import random
import matplotlib.pyplot as plt

def calculate_projection_matrix(image, markers):
    """
    To solve for the projection matrix. You need to set up a system of
    equations using the corresponding 2D and 3D points. See the handout, Q5
    of the written questions, or the lecture slides for how to set up these
    equations.

    Don't forget to set M_34 = 1 in this system to fix the scale.

    :param image: a single image in our camera system
    :param markers: dictionary of markerID to 4x3 array containing 3D points
    
    :return: M, the camera projection matrix which maps 3D world coordinates
    of provided aruco markers to image coordinates
             residual, the error in the estimation of M given the point sets
    """
    ######################
    # Do not change this #
    ######################

    # Markers is a dictionary mapping a marker ID to a 4x3 array
    # containing the 3d points for each of the 4 corners of the
    # marker in our scanning setup
    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_1000)
    parameters = cv2.aruco.DetectorParameters_create()

    markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(
        image, dictionary, parameters=parameters)
    markerIds = [m[0] for m in markerIds]
    markerCorners = [m[0] for m in markerCorners]

    points2d = []
    points3d = []

    for markerId, marker in zip(markerIds, markerCorners):
        if markerId in markers:
            for j, corner in enumerate(marker):
                points2d.append(corner)
                points3d.append(markers[markerId][j])

    points2d = np.array(points2d)
    points3d = np.array(points3d)

    ########################
    # TODO: Your code here #
    ########################
    # # Placeholder values. This M matrix came from a call to rand(3,4). It leads to a high residual.
    residual = 7 # Arbitrary stencil code initial value placeholder

    b = points2d.flatten()
    ones = np.ones((len(points3d), 1))
    matvals = np.concatenate((points3d,ones),axis=1)
    zeros = np.zeros_like(matvals)
    evenrows = np.concatenate((matvals, zeros),axis=1)
    oddrows = np.concatenate((zeros, matvals),axis=1)
    a = np.repeat(oddrows,2,axis=0)
    
    for i in range(0,len(evenrows)):
        a[2*i] = evenrows[i]

    mat3 = np.repeat(points3d,2,axis=0)
    mat3 = mat3*b[:,np.newaxis]
    mat3 = mat3*-1

    a = np.concatenate((a, mat3),axis=1)

    M = np.append(np.linalg.lstsq(a,b)[0],[1])
    M = np.array(M)
    M = M.reshape((3,4))
    residual = np.linalg.lstsq(a,b)[1]

    return M, residual

def normalize_coordinates(points):
    """
    ============================ EXTRA CREDIT ============================
    Normalize the given Points before computing the fundamental matrix. You
    should perform the normalization to make the mean of the points 0
    and the average magnitude 1.0.

    The transformation matrix T is the product of the scale and offset matrices.

    Offset Matrix
    Find c_u and c_v and create a matrix of the form in the handout for T_offset

    Scale Matrix
    Subtract the means of the u and v coordinates, then take the reciprocal of
    their standard deviation i.e. 1 / np.std([...]). Then construct the scale
    matrix in the form provided in the handout for T_scale

    :param points: set of [n x 2] 2D points
    :return: a tuple of (normalized_points, T) where T is the [3 x 3] transformation
    matrix
    """
    ########################
    # TODO: Your code here #
    ########################
    # This is a placeholder with the identity matrix for T replace with the
    # real transformation matrix for this set of points
    T = np.eye(3)

    return points, T

def estimate_fundamental_matrix(points1, points2):
    """
    Estimates the fundamental matrix given set of point correspondences in
    points1 and points2. The fundamental matrix will transform a point into 
    a line within the second image - the epipolar line - such that F x' = l. 
    Fitting a fundamental matrix to a set of points will try to minimize the 
    error of all points x to their respective epipolar lines transformed 
    from x'. The residual can be computed as the difference from the known 
    geometric constraint that x^T F x' = 0.

    points1 is an [n x 2] matrix of 2D coordinate of points on Image A
    points2 is an [n x 2] matrix of 2D coordinate of points on Image B

    Implement this function efficiently as it will be
    called repeatedly within the RANSAC part of the project.

    If you normalize your coordinates for extra credit, don't forget to adjust
    your fundamental matrix so that it can operate on the original pixel
    coordinates!

    :return F_matrix, the [3 x 3] fundamental matrix
            residual, the error in the estimation
    """
    ########################
    # TODO: Your code here #
    ########################
    funmat = []
    for i in len(points1):
        u = points1[i][0]
        v = points1[i][1]
        up = points2[i][0]
        vp = points2[i][1]
        uup = u * up
        uvp = u * vp
        vup = v * up
        vvp = v * vp
        pmat = []
        pmat.append(uup)
        pmat.append(uvp)
        pmat.append(u)
        pmat.append(vup)
        pmat.append(vvp)
        pmat.append(v)
        pmat.append(up)
        pmat.append(vp)
        funmat.append(pmat)

    funmat = np.array(funmat)
    zeros = np.ones(len(points1))

    F_matrix = np.append(np.linalg.lstsq(funmat,zeros)[0],-1)
    F_matrix = np.array(F_matrix)
    F_matrix = F_matrix.reshape(3,3)
    sumsq = F_matrix[0][0]*F_matrix[0][0] + F_matrix[0][1]*F_matrix[0][1] + F_matrix[0][2]*F_matrix[0][2] + F_matrix[1][0]*F_matrix[1][0] + F_matrix[1][1]*F_matrix[1][1] + F_matrix[1][2]*F_matrix[1][2 + F_matrix[2][0]*F_matrix[2][0] + F_matrix[2][1]*F_matrix[2][1] + F_matrix[2][2]*F_matrix[2][2]
    residual = np.sqrt(sumsq)

    return F_matrix, residual

def ransac_fundamental_matrix(matches1, matches2, num_iters):
    """
    Implement RANSAC to find the best fundamental matrix robustly
    by randomly sampling interest points.
    
    Inputs:
    matches1 and matches2 are the [N x 2] coordinates of the possibly
    matching points across two images. Each row is a correspondence
     (e.g. row 42 of matches1 is a point that corresponds to row 42 of matches2)

    Outputs:
    best_Fmatrix is the [3 x 3] fundamental matrix
    best_inliers1 and best_inliers2 are the [M x 2] subset of matches1 and matches2 that
    are inliners with respect to best_Fmatrix
    best_inlier_residual is the error induced by best_Fmatrix

    :return: best_Fmatrix, inliers1, inliers2, best_inlier_residual
    """
    # DO NOT TOUCH THE FOLLOWING LINES
    random.seed(0)
    np.random.seed(0)
    
    ########################
    # TODO: Your code here #
    ########################

    # Your RANSAC loop should contain a call to your 'estimate_fundamental_matrix()'

    # Placeholder values
    best_Fmatrix = np.zeros((3,3))
    best_inliers_a = []
    best_inliers_b = []
    best_inlier_residual = 5 # Arbitrary stencil code initial value placeholder.

    maxm = 0
    threshold = 0.0025
    
    for i in range(num_iters):
        index = np.random.choice(len(matches1),9,replace=False)
        m1 = matches1[index]
        m2 = matches2[index]
        funmat = estimate_fundamental_matrix(m1,m2)
        count = 0
        a = []
        b = []
        for i in len(matches1):
            mat1 = matches1[i]
            mat2 = matches2[i]
            match1 = np.append(mat1,1)
            match2 = np.append(mat2,1)
            mult = np.abs(np.matmul(np.matmul(match2.T, funmat), match1))
            if threshold > mult:
                count+=1
                np.append(inlier_counts, count)
                np.append(inlier_residuals, mult)
                a.append(mat1)
                b.append(mat2)
        
        if count > maxm:
            best_Fmatrix = funmat
            best_inliers_a = a
            best_inliers_b = b
            best_inlier_residual = inlier_residuals

    best_Fmatrix = np.array(best_Fmatrix)
    best_inliers_a = np.array(best_inliers_a)
    best_inliers_b = np.array(best_inliers_b)
    best_inlier_residual = np.array(best_inlier_residual)

    # For your report, we ask you to visualize RANSAC's 
    # convergence over iterations. 
    # For each iteration, append your inlier count and residual to the global variables:
    #   inlier_counts = []
    #   inlier_residuals = []
    # Then add flag --visualize-ransac to plot these using visualize_ransac()
    

    return best_Fmatrix, best_inliers_a, best_inliers_b, best_inlier_residual

def matches_to_3d(points2d_1, points2d_2, M1, M2, threshold=1.0):
    """
    Given two sets of corresponding 2D points and two projection matrices, you will need to solve
    for the ground-truth 3D points using np.linalg.lstsq().

    You may find that some 3D points have high residual/error, in which case you 
    can return a subset of the 3D points that lie within a certain threshold.
    In this case, also return subsets of the initial points2d_1, points2d_2 that
    correspond to this new inlier set. You may modify the default value of threshold above.
    All local helper code that calls this function will use this default value, but we
    will pass in a different value when autograding.

    N is the input number of point correspondences
    M is the output number of 3D points / inlier point correspondences; M could equal N.

    :param points2d_1: [N x 2] points from image1
    :param points2d_2: [N x 2] points from image2
    :param M1: [3 x 4] projection matrix of image1
    :param M2: [3 x 4] projection matrix of image2
    :param threshold: scalar value representing the maximum allowed residual for a solved 3D point

    :return points3d_inlier: [M x 3] NumPy array of solved ground truth 3D points for each pair of 2D
    points from points2d_1 and points2d_2
    :return points2d_1_inlier: [M x 2] points as subset of inlier points from points2d_1
    :return points2d_2_inlier: [M x 2] points as subset of inlier points from points2d_2
    """
    ########################
    # TODO: Your code here #

    # Initial random values for 3D points
    points3d_inlier = np.random.rand(len(points2d_1), 3)
    points2d_1_inlier = np.array(points2d_1, copy=True) # only modify if using threshold
    points2d_2_inlier = np.array(points2d_2, copy=True) # only modify if using threshold

    # Solve for ground truth points

    ########################

    return points3d_inlier, points2d_1_inlier, points2d_2_inlier


#/////////////////////////////DO NOT CHANGE BELOW LINE///////////////////////////////
inlier_counts = []
inlier_residuals = []

def visualize_ransac():
    iterations = np.arange(len(inlier_counts))
    best_inlier_counts = np.maximum.accumulate(inlier_counts)
    best_inlier_residuals = np.minimum.accumulate(inlier_residuals)

    plt.figure(1, figsize = (8, 8))
    plt.subplot(211)
    plt.plot(iterations, inlier_counts, label='Current Inlier Count', color='red')
    plt.plot(iterations, best_inlier_counts, label='Best Inlier Count', color='blue')
    plt.xlabel("Iteration")
    plt.ylabel("Number of Inliers")
    plt.title('Current Inliers vs. Best Inliers per Iteration')
    plt.legend()

    plt.subplot(212)
    plt.plot(iterations, inlier_residuals, label='Current Inlier Residual', color='red')
    plt.plot(iterations, best_inlier_residuals, label='Best Inlier Residual', color='blue')
    plt.xlabel("Iteration")
    plt.ylabel("Residual")
    plt.title('Current Residual vs. Best Residual per Iteration')
    plt.legend()
    plt.show()
