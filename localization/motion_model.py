import numpy as np

class MotionModel:

    def __init__(self, node):
        ####################################
        # TODO
        # Do any precomputation for the motion
        # model here.

        pass

        ####################################
            
    def x_to_T(self, x):
        """
        Transform pose vector into transformation matrix

        args:
            x: A 3-vector [x y theta]

        return:
            T: 3x3 transformation matrix
        """
        th = x[2]
        R = [np.cos(th), -np.sin(th), np.sin(th), np.cos(th)]
        T = np.array([[R[0], R[1], x[0]],[R[2], R[3], x[1]],[0, 0, 1]])
        return T


    def evaluate(self, particles, odometry):
        """
        Update the particles to reflect probable
        future states given the odometry data.

        args:
            particles: An Nx3 matrix of the form:

                [x0 y0 theta0]
                [x1 y1 theta1]
                [    ...     ]

            odometry: A 3-vector [dx dy dtheta]

        returns:
            particles: An updated matrix of the
                same size
        """

        ####################################
        # TODO
        e_x  = 0.02          ## last best: 0.02, 0.01, 0.001
        e_y  = 0.01
        e_th = 0.001
        noise = np.array([np.random.normal(odometry[0], e_x,  len(particles)),
                          np.random.normal(odometry[1], e_y,  len(particles)),
                          np.random.normal(odometry[2], e_th, len(particles))])
        noise = noise.T
        for i in range(len(particles)):
            T  = self.x_to_T(particles[i])
            dT = self.x_to_T(noise[i])
            T_new = T @ dT
            particles[i] = [T_new[0][2], T_new[1][2], np.arctan2(T_new[1][0], T_new[0][0])]
            # p[i] = [T[0][2], T[1][2], np.arccos(T[0][0])] # Check original particle values
        return particles

        ####################################