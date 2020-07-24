#!/usr/bin/env python

"""ground_truth_republisher.py

Republishes a corrected version of the raw ground truth state estimate from the simulation in the
same frame as the state estimator.
Gives the option to republish onto another topic linear velocity in the frame of the vehicle.
"""

import numpy as np
import rospy
import tf
from geometry_msgs.msg import TransformStamped, Vector3Stamped
from nav_msgs.msg import Odometry
from tf2_geometry_msgs import do_transform_vector3


class GroundTruthRepublisher(object):

    def __init__(self, namespace='ground_truth_republisher'):
        """Initialize this _ground_truth_republisher"""
        # topic name specified inside autoRallyPlatform.urdf.xacro
        ground_truth_raw_topic = "/ground_truth/state_raw"

        rospy.init_node(namespace, anonymous=True)
        self.pub = rospy.Publisher("/ground_truth/state", Odometry, queue_size=1)
        self.sub = rospy.Subscriber(ground_truth_raw_topic, Odometry, self.handle_pose)

        if rospy.get_param(namespace + "/transform_to_vehicle_frame", False):
            self.pub_2 = rospy.Publisher("/ground_truth/state_transformed", Odometry, queue_size=1)
            self.sub_2 = rospy.Subscriber(ground_truth_raw_topic, Odometry, self.transform_velocity)

    def transform_velocity(self, msg):
        # init the vector object for linear velocity
        twist_vel = Vector3Stamped()

        # get the linear velocity from the msg
        linear = msg.twist.twist.linear

        twist_vel.vector.x = linear.x
        twist_vel.vector.y = linear.y
        twist_vel.vector.z = linear.z

        # init the transform object
        t = TransformStamped()

        # get the orientation from the msg
        orientation = msg.pose.pose.orientation

        # ground truth gives the orientation of the body frame so need inverse of body -> global transformation
        # it inverts the components x,y,z of orientation by flipping the sign
        q = tf.transformations.quaternion_inverse([orientation.x, orientation.y, orientation.z, orientation.w])

        t.transform.rotation.x = q[0]
        t.transform.rotation.y = q[1]
        t.transform.rotation.z = q[2]
        t.transform.rotation.w = q[3]

        # transform the velocity to the vehicle frame
        res = do_transform_vector3(twist_vel, t)

        # update message
        msg.twist.twist.linear = res.vector

        self.pub_2.publish(msg)

    # this is the callback
    def handle_pose(self, msg):
        # set frame to be the same as state estimator output
        msg.header.frame_id = 'odom'
        msg.child_frame_id = 'base_link'

        rotationNeg90Deg = np.array([[0, -1], [1, 0]])

        # print msg.pose.pose.position.x, msg.pose.pose.position.y

        # rotate position 90 deg around z-axis
        pos = np.dot(rotationNeg90Deg, np.array([msg.pose.pose.position.x, msg.pose.pose.position.y]))
        msg.pose.pose.position.x = pos[0]
        msg.pose.pose.position.y = pos[1]

        # rotate orientation 90 deg around z-axis
        q_rot = tf.transformations.quaternion_from_euler(0, 0, 1.57)
        q_new = tf.transformations.quaternion_multiply(q_rot, [msg.pose.pose.orientation.x, msg.pose.pose.orientation.y,
                                                               msg.pose.pose.orientation.z,
                                                               msg.pose.pose.orientation.w])

        msg.pose.pose.orientation.x = q_new[0]
        msg.pose.pose.orientation.y = q_new[1]
        msg.pose.pose.orientation.z = q_new[2]
        msg.pose.pose.orientation.w = q_new[3]

        # rotate linear velocity 90 deg around z-axis
        lin = np.dot(rotationNeg90Deg, np.array([msg.twist.twist.linear.x, msg.twist.twist.linear.y]))
        msg.twist.twist.linear.x = lin[0]
        msg.twist.twist.linear.y = lin[1]

        self.pub.publish(msg)


if __name__ == "__main__":
    ground_truth_republisher = GroundTruthRepublisher()
    rospy.spin()
