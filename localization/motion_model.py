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
        N = len(particles)
        e_x  = 0.015          ## last best: 0.02, 0.01, 0.001
        e_y  = 0.01
        e_th = 0.001

        noise = np.array([np.random.normal(odometry[0], e_x,  len(particles)),
                          np.random.normal(odometry[1], e_y,  len(particles)),
                          np.random.normal(odometry[2], e_th, len(particles))])
        noise = noise.T
        x  = np.array(particles)
        dx = np.array(noise)
        T  = np.array([[np.cos( x[:,2]), -np.sin( x[:,2]),  x[:,0]],[np.sin( x[:,2]), np.cos( x[:,2]),  x[:,1]],[np.zeros(N), np.zeros(N), np.ones(N)]])
        dT = np.array([[np.cos(dx[:,2]), -np.sin(dx[:,2]), dx[:,0]],[np.sin(dx[:,2]), np.cos(dx[:,2]), dx[:,1]],[np.zeros(N), np.zeros(N), np.ones(N)]])

        p_x = T[0,0,:]*dT[0,2,:]+T[0,1,:]*dT[1,2,:]+T[0,2,:]*dT[2,2,:]
        p_y = T[1,0,:]*dT[0,2,:]+T[1,1,:]*dT[1,2,:]+T[1,2,:]*dT[2,2,:]
        cos_th = T[0,0,:]*dT[0,0,:]+T[0,1,:]*dT[1,0,:]+T[0,2,:]*dT[2,0,:]
        sin_th = T[1,0,:]*dT[0,0,:]+T[1,1,:]*dT[1,0,:]+T[1,2,:]*dT[2,0,:]
        p_th = np.arctan2(sin_th, cos_th)
        
        return np.stack((p_x,p_y,p_th), axis=1)

        ####################################
