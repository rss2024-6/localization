import numpy as np

class MotionModel:

    def __init__(self, node):
        ####################################
        # TODO
        # Do any precomputation for the motion
        # model here.

        pass

        ####################################
            
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
        e_x  = 0.015          ## last best: 0.02, 0.01, 0.001
        e_y  = 0.01
        e_th = 0.001
        noise = np.array([np.random.normal(odometry[0], e_x,  len(particles)),
                          np.random.normal(odometry[1], e_y,  len(particles)),
                          np.random.normal(odometry[2], e_th, len(particles))])
        noise = noise.T
        for i in range(len(particles)):
            x  = particles[i]
            dx = noise[i]
            T  = np.array([[np.cos( x[2]), -np.sin( x[2]),  x[0]],[np.sin( x[2]), np.cos( x[2]),  x[1]],[0, 0, 1]])
            dT = np.array([[np.cos(dx[2]), -np.sin(dx[2]), dx[0]],[np.sin(dx[2]), np.cos(dx[2]), dx[1]],[0, 0, 1]])
            T_new = T @ dT
            particles[i] = [T_new[0][2], T_new[1][2], np.arctan2(T_new[1][0], T_new[0][0])]
        return particles

        ####################################
