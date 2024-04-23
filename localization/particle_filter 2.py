from localization.sensor_model import SensorModel
from localization.motion_model import MotionModel

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped, Quaternion, TransformStamped
from tf_transformations import euler_from_quaternion, quaternion_from_euler
import tf2_ros
from sklearn.cluster import DBSCAN

import numpy as np

from rclpy.node import Node
import rclpy
import math


class ParticleFilter(Node):

    def __init__(self):
        super().__init__("particle_filter")

        self.declare_parameter('particle_filter_frame', "default")
        self.particle_filter_frame = self.get_parameter('particle_filter_frame').get_parameter_value().string_value
        self.simulation = True

        #  *Important Note #1:* It is critical for your particle
        #     filter to obtain the following topic names from the
        #     parameters for the autograder to work correctly. Note
        #     that while the Odometry message contains both a pose and
        #     a twist component, you will only be provided with the
        #     twist component, so you should rely only on that
        #     information, and *not* use the pose component.
        
        self.declare_parameter('odom_topic', "/odom")
        self.declare_parameter('scan_topic', "/scan")

        scan_topic = self.get_parameter("scan_topic").get_parameter_value().string_value
        odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value

        self.laser_sub = self.create_subscription(LaserScan, scan_topic,
                                                  self.laser_callback,
                                                  1)

        self.odom_sub = self.create_subscription(Odometry, odom_topic,
                                                 self.odom_callback,
                                                 1)

        #  *Important Note #2:* You must respond to pose
        #     initialization requests sent to the /initialpose
        #     topic. You can test that this works properly using the
        #     "Pose Estimate" feature in RViz, which publishes to
        #     /initialpose.

        self.pose_sub = self.create_subscription(PoseWithCovarianceStamped, "/initialpose",
                                                 self.pose_callback,
                                                 1)

        #  *Important Note #3:* You must publish your pose estimate to
        #     the following topic. In particular, you must use the
        #     pose field of the Odometry message. You do not need to
        #     provide the twist part of the Odometry message. The
        #     odometry you publish here should be with respect to the
        #     "/map" frame.

        self.odom_pub = self.create_publisher(Odometry, "/pf/pose/odom", 1)

        # Initialize the models
        self.motion_model = MotionModel(self)
        self.sensor_model = SensorModel(self)

        self.get_logger().info("=============+READY+=============")

        # Implement the MCL algorithm
        # using the sensor model and the motion model
        #
        # Make sure you include some way to initialize
        # your particles, ideally with some sort
        # of interactive interface in rviz
        #
        # Publish a transformation frame between the map
        # and the particle_filter_frame.

        self.particles_len = 500
        self.particles = np.empty(self.particles_len, 3) # TODO initialize particles array properly
        self.initial_pose = PoseWithCovarianceStamped()
        self.previous_pose = None
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

    # sets the initial pose from rviz
    def pose_callback(self, msg):
        self.initial_pose = msg
        quaternion = self.initial_pose.pose.pose.orientation
        theta = euler_from_quaternion([quaternion.x, quaternion.y, quaternion.z, quaternion.w])[2]
        self.particles = np.random.normal([msg.pose.pose.position.x, msg.pose.pose.position.y, theta], 0.1, (self.particles_len, 3))


    def laser_callback(self, msg):
        observation = msg.ranges
        probabilities = self.sensor_model.evaluate(self.particles, observation)
        # resample particles
        indicies = np.random.choice(len(self.particles), size=len(self.particles), p=probabilities)
        self.particles = [self.particles[i] for i in indicies]
        self.publish_average_pose()


    def odom_callback(self, msg):
        twist = msg.twist.twist 
        prev_twist = self.previous_pose.twist.twist

        if self.previous_pose is not None:
            vx = (twist.linear.x + prev_twist.linear.x) / 2
            vy = (twist.linear.y + prev_twist.linear.y) / 2
            omega = (twist.linear.z + prev_twist.linear.z) / 2

            dt = (msg.header.stamp.sec - self.previous_pose.header.stamp.sec) + (msg.header.stamp.nanosec - self.previous_pose.header.stamp.nanosec) * 1e-9

            dx = vx * dt
            dy = vy * dt
            dTheta = omega * dt
            
            odom_data = [dx, dy, dTheta]

            self.particles = self.motion_model.evaluate(self.particles, odom_data)
        else:
            self.previous_pose = msg
        
        self.publish_average_pose()

    def circular_mean(self, radians):
        # Calculate the sum of sin and cos values
        sin_sum = sum([math.sin(rad) for rad in radians])
        cos_sum = sum([math.cos(rad) for rad in radians])

        # Calculate the circular mean using arctan2
        mean_rad = math.atan2(sin_sum, cos_sum)

        return mean_rad
    
    # find averge pose based on the particles
    def publish_average_pose(self):
        # avg_x = np.mean([p[0] for p in self.particles])
        # avg_y = np.mean([p[1] for p in self.particles])

        # avg_x = np.median([p[0] for p in self.particles])
        # avg_y = np.median([p[1] for p in self.particles])
        # avg_theta = self.circular_mean([p[2] for p in self.particles])

        avg_pose = self.get_avg_pose(self.particles)
        avg_x = avg_pose[0]
        avg_y = avg_pose[1]
        avg_theta = avg_pose[2]
        # avg_theta = self.circular_mean([p[2] for p in self.particles])

        odom = Odometry()
        odom.header.stamp = self.get_clock().now().to_msg()
        odom.header.frame_id = "map"
        odom.child_frame_id = self.particle_filter_frame
        odom.pose.pose.position.x = avg_x
        odom.pose.pose.position.y = avg_y
        odom.pose.pose.position.z = 0.0
        
        q = quaternion_from_euler(0, 0, avg_theta)
        quaternion = Quaternion()
        quaternion.x = q[0]
        quaternion.y = q[1]
        quaternion.z = q[2]
        quaternion.w = q[3]

        odom.pose.pose.orientation = quaternion

        self.odom_pub.publish(odom)
        
        #publish transform
        trans_msg = TransformStamped()
        trans_msg.header.stamp = self.get_clock().now().to_msg()
        trans_msg.header.frame_id = "map"

        if self.simulation:
            trans_msg.child_frame_id = "base_link_pf"
        else:
            trans_msg.child_frame_id = "base_link"
        
        trans_msg.transform.translation.x = avg_x
        trans_msg.transform.translation.y = avg_y
        trans_msg.transform.translation.z = 0.0
        trans_msg.transform.rotation = quaternion

        self.tf_broadcaster.sendTransform(trans_msg)
    
    def get_avg_pose(self, particles): #Mode clustering
        dbscan = DBSCAN(eps=1, min_samples=10)
        clusters = dbscan.fit_predict(particles[:, :2])  # Only x and y coordinates are considered for clustering

        # Finding the cluster with the highest number of particles
        largest_cluster_id = np.argmax(np.bincount(clusters[clusters >= 0]))

        # Filtering particles that belong to the largest cluster
        largest_cluster_particles = particles[clusters == largest_cluster_id]

        # Calculating the average pose (x, y, and circular mean for theta) of the largest cluster
        average_pose = np.mean(largest_cluster_particles[:, :2], axis=0)  # Mean x and y
        average_theta = self.circular_mean(largest_cluster_particles[:, 2])

        return average_pose[0], average_pose[1], average_theta


        



def main(args=None):
    rclpy.init(args=args)
    pf = ParticleFilter()
    rclpy.spin(pf)
    rclpy.shutdown()