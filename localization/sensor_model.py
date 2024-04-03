import numpy as np
from scan_simulator_2d import PyScanSimulator2D
# Try to change to just `from scan_simulator_2d import PyScanSimulator2D` 
# if any error re: scan_simulator_2d occurs

from tf_transformations import euler_from_quaternion

from nav_msgs.msg import OccupancyGrid
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys

np.set_printoptions(threshold=sys.maxsize)


class SensorModel:

    def __init__(self): #, node

        node.declare_parameter('map_topic', "default")
        node.declare_parameter('num_beams_per_particle', "default")
        node.declare_parameter('scan_theta_discretization', "default")
        node.declare_parameter('scan_field_of_view', "default")
        node.declare_parameter('lidar_scale_to_map_scale', 1)

        self.map_topic = node.get_parameter('map_topic').get_parameter_value().string_value
        self.num_beams_per_particle = node.get_parameter('num_beams_per_particle').get_parameter_value().integer_value
        self.scan_theta_discretization = node.get_parameter(
            'scan_theta_discretization').get_parameter_value().double_value
        self.scan_field_of_view = node.get_parameter('scan_field_of_view').get_parameter_value().double_value
        self.lidar_scale_to_map_scale = node.get_parameter(
            'lidar_scale_to_map_scale').get_parameter_value().double_value

        self.resolution = 1.0
        self.lidar_scale_to_map_scale = 1.0
        ####################################
        # Adjust these parameters
        self.alpha_hit = 0.74
        self.alpha_short = 0.07
        self.alpha_max = 0.07
        self.alpha_rand = 0.12
        self.sigma_hit = 8.0

        # Your sensor table will be a `table_width` x `table_width` np array:
        self.table_width = 201
        ####################################

        node.get_logger().info("%s" % self.map_topic)
        node.get_logger().info("%s" % self.num_beams_per_particle)
        node.get_logger().info("%s" % self.scan_theta_discretization)
        node.get_logger().info("%s" % self.scan_field_of_view)

        # Precompute the sensor model table
        self.sensor_model_table = np.empty((self.table_width, self.table_width))
        self.precompute_sensor_model()

        # Create a simulated laser scan
        self.scan_sim = PyScanSimulator2D(
            self.num_beams_per_particle,
            self.scan_field_of_view,
            0,  # This is not the simulator, don't add noise
            0.01,  # This is used as an epsilon
            self.scan_theta_discretization)

        # Subscribe to the map
        self.map = None
        self.map_set = False

        self.map_subscriber = node.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self.map_callback,
            1)

    def calc_score(self, z_k, d):
        z_max = 200
        n = 1
        p_hit = 0
        p_short = 0
        p_max = 0
        p_rand = 0
        #epsilon = 0.1
        if z_k >= 0 and z_k <= z_max:
            p_hit = n*1/(np.sqrt(2*np.pi*self.sigma_hit**2))*np.exp(-(z_k-d)**2/(2*self.sigma_hit**2))

        if z_k >= 0 and z_k <= d and d != 0:
            p_short = 2/d*(1-z_k/d)

        if z_k == z_max:
            p_max = 1

        if z_k >= 0 and z_k <= z_max:
            p_rand = 1/z_max

        return (p_hit, p_short, p_max, p_rand)
    
    def precompute_sensor_model(self):
        """
        Generate and store a table which represents the sensor model.

        For each discrete computed range value, this provides the probability of 
        measuring any (discrete) range. This table is indexed by the sensor model
        at runtime by discretizing the measurements and computed ranges from
        RangeLibc.
        This table must be implemented as a numpy 2D array.

        Compute the table based on class parameters alpha_hit, alpha_short,
        alpha_max, alpha_rand, sigma_hit, and table_width.

        args:
            N/A

        returns:
            No return type. Directly modify `self.sensor_model_table`.
        """

        phits = np.zeros(self.sensor_model_table.shape)
        extra = np.zeros(self.sensor_model_table.shape)

        for zk in range(self.table_width):
            for d in range(self.table_width):
                p_hit, p_short, p_max, p_rand = self.calc_score(zk, d)
                phits[zk][d] = p_hit
                extra[zk][d] = self.alpha_short*p_short + self.alpha_max*p_max + self.alpha_rand*p_rand

        # Normalize p_hit
        nfac = np.sum(phits, axis = 0).reshape(phits.shape[1], 1)
        print(nfac.shape)

        phits /= nfac
        
        self.sensor_model_table = self.alpha_hit*phits + extra
        self.sensor_model_table /= np.sum(self.sensor_model_table, axis = 0)

        A = self.sensor_model_table
        x = np.arange(A.shape[1])
        y = np.arange(A.shape[0])
        X, Y = np.meshgrid(x, y)

        # Create a 3D plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the 3D surface
        surf = ax.plot_surface(X, Y, A, cmap='viridis')

        # Add labels and title
        ax.set_xlabel('True Distance')
        ax.set_ylabel('Measured Distance')
        ax.set_zlabel('Probability Score')
        plt.title('Probability Manifold (3D Surface)')

        ax.invert_xaxis()
        ax.set_zlim(0, 0.12)

        # Add colorbar
        plt.colorbar(surf, label='Probability Score')

        # Show the plot
        plt.show()

    def evaluate(self, particles, observation):
        """
        Evaluate how likely each particle is given
        the observed scan.

        args:
            particles: An Nx3 matrix of the form:

                [x0 y0 theta0]
                [x1 y0 theta1]
                [    ...     ]

            observation: A vector of lidar data measured
                from the actual lidar. THIS IS Z_K. Each range in Z_K is Z_K^i

        returns:
           probabilities: A vector of length N representing
               the probability of each particle existing
               given the observation and the map.
        """

        if not self.map_set:
            return

        ####################################
        # TODO
        # Evaluate the sensor model here!
        #
        # You will probably want to use this function
        # to perform ray tracing from all the particles.
        # This produces a matrix of size N x num_beams_per_particle 

        # 0. Downsample LIDAR data for efficiency
        inds = list(np.round(np.linspace(0, len(observation) - 1, self.num_beams_per_particle)))
        obs_ds = observation[inds]

        # 1. Convert LIDAR data to pixels (round to nearest integer)
        obs_px = np.round(obs_ds / (self.resolution * self.lidar_scale_to_map_scale))

        # 2. Get scans from particle POV
        scans = self.scan_sim.scan(particles)

        # 3. Convert scans to pixels
        scans_px = scans / (self.resolution * self.lidar_scale_to_map_scale)

        # 4. Get P(obs | particle x_k) using precomputed table
        probabilities = np.zeros(len(particles))

        for i in range(len(particles)):
            ds = scans_px[i]
            probabilities[i] = np.prod(np.array([self.sensor_model_table[obs_px[k]][ds[k]] for k in range(self.num_beams_per_particle)]))
        
        return probabilities

        ####################################

    def map_callback(self, map_msg):
        # Convert the map to a numpy array
        self.map = np.array(map_msg.data, np.double) / 100.
        self.map = np.clip(self.map, 0, 1)

        self.resolution = map_msg.info.resolution

        # Convert the origin to a tuple
        origin_p = map_msg.info.origin.position
        origin_o = map_msg.info.origin.orientation
        origin_o = euler_from_quaternion((
            origin_o.x,
            origin_o.y,
            origin_o.z,
            origin_o.w))
        origin = (origin_p.x, origin_p.y, origin_o[2])

        # Initialize a map with the laser scan
        self.scan_sim.set_map(
            self.map,
            map_msg.info.height,
            map_msg.info.width,
            map_msg.info.resolution,
            origin,
            0.5)  # Consider anything < 0.5 to be free

        # Make the map set
        self.map_set = True

        print("Map initialized")

# s = SensorModel()
