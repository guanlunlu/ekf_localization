#!/usr/bin/env python3
import rospy
import math
import numpy as np
import tf
import operator
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from obstacle_detector.msg import Obstacles
from tf.transformations import quaternion_from_euler
from tf.transformations import euler_from_quaternion
pi = math.pi

class state_vector:
    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta
    
    def matrix(self):
        return np.mat([[self.x], 
                       [self.y],
                       [self.theta]])

    def show(self):
        return ((self.x, self.y, self.theta))

class update_feature:
    def __init__(self, j, H, S, z, z_hat, j_threshold):
        self.j = j
        self.H = H
        self.H_t = np.transpose(H)
        self.S = S
        self.S_inv = np.linalg.inv(S)
        self.z = z
        self.z_hat = z_hat
        self.j_threshold = j_threshold
    
    def update(self, state_pre, cov_pre):
        updated = 0
        # print("j = ", self.j)
        # print("j_thres", self.j_threshold)

        if self.j > self.j_threshold:
            K = cov_pre * self.H_t * self.S_inv
            delta_z = self.z - self.z_hat
            # ensure delta_z phi in the domain of [-pi, pi]
            delta_phi = self.theta_convert(delta_z[1,0])
            delta_r = delta_z[0,0]
            delta_z = np.mat([[delta_r],
                            [delta_phi], 
                            [0]])
            mean_est_matrix = state_pre.matrix() + K * delta_z
            mean_est_theta = self.theta_convert(mean_est_matrix[2,0])
            mean_est = state_vector(mean_est_matrix[0,0], 
                                    mean_est_matrix[1,0],
                                    mean_est_theta)
            I = (3,3)
            I = np.ones(I)
            cov_est = (I - K * self.H) * cov_pre
            # print("MEASUREMENT UPDATE AVAILABLE !!!")
            # print("j = ", self.j)
            # print("mean estimated = ", (mean_est.x, mean_est.y, mean_est.theta))
            updated = 1
        else:
            # print("MEASUREMENT UPDATE NOT AVAILABLE !!!")
            # print((state_pre.x, state_pre.y, state_pre.theta))
            mean_est = state_pre
            cov_est = cov_pre
        
        return (mean_est, cov_est, updated)

    def theta_convert(self, input):
        # convert rad domain to [-pi, pi]
        pi = math.pi
        if input >=0:
            input = math.fmod(input, 2*pi)
            if input > pi:
                input -= 2*pi
                output = input
            else:
                output = input
        else:
            input *= -1
            input = math.fmod(input, 2*pi)
            if input > pi:
                input -= 2*pi
                output = input*-1
            else:
                output = input*-1
        return output

class localization:
    def __init__(self) -> None:
        # self.motion_sub = rospy.Subscriber("cmd_vel", Twist, self.motionCallback)
        self.odom_sub = rospy.Subscriber("odom", Odometry, self.odomCallback)
        self.tf_broadcaster = tf.TransformBroadcaster()
        self.measurement_sub = rospy.Subscriber("raw_obstacles", Obstacles, self.measurementCallback)
        self.prediction_pub = rospy.Publisher("ekf_prediction", PoseWithCovarianceStamped, queue_size=10)
        self.estimation_pub = rospy.Publisher("ekf_estimation", PoseWithCovarianceStamped, queue_size=10)
        self.tf_listener = tf.TransformListener()
        self.frequency = 50
        # self.d_t = 1/self.control_frequency
        
        self.odom_init = 0
        self.init_pose = state_vector(0.5 ,0.5 ,pi/2)
        self.odom_past = self.init_pose
        self.State_past = self.init_pose

        self.Cov_past = np.mat([[0, 0, 0],
                                [0, 0, 0],
                                [0, 0, 0]])

        self.Model_const = np.mat([[0.4, 0, 0],
                                   [0, 0.4, 0],
                                   [0, 0, 0.4]])

        # sensing uncertainty
        self.Q = ([[0.05, 0, 0],
                   [0, 0.05, 0],
                   [0, 0, 0.05]])
        self.measurement_update_threshold = 0.8
        self.observed_features = []
        self.landmark1 = state_vector(1, -0.092, 0)
        self.landmark2 = state_vector(0.055, 3.1, 0)
        self.landmark3 = state_vector(1.953, 3.1, 0)
        self.landmark_list = [self.landmark1, self.landmark2, self.landmark3]

        self.odom_bf = state_vector(0,0,0)
        self.map_odom = state_vector(0,0,0)

        self.stamp_past = rospy.get_time()
        self.d_t = 0

    def odomCallback(self, odom):
        stamp = rospy.get_time()
        # rospy.loginfo("odom time: %s, now: %s", odom.header.stamp, stamp)
        v_x = odom.twist.twist.linear.x
        v_y = odom.twist.twist.linear.y
        w = odom.twist.twist.angular.z
        U_t = state_vector(v_x, v_y, w)
        self.d_t = stamp - self.stamp_past
        state_pre, cov_pre = self.state_prediction(self.State_past, self.Cov_past, U_t)
        print("v_x, v_y, w = ", v_x, v_y, w)
        self.State_past = state_pre
        self.Cov_past = cov_pre
        # print("state_pre = \n", state_pre.matrix())
        # print("cov_pre = \n", cov_pre)
        # print("------------------------------")
        # self.ekf_localization(v_x, v_y, w)
        # self.publish_ekf_pose(stamp + rospy.Duration(0.2))
        # self.broadcast_ekf_pos_tf(odom)
        self.stamp_past = stamp
        rospy.Rate(60).sleep()

    def measurementCallback(self, data):
        obstacleList = []
        for i in data.circles:
            obstacleList.append(state_vector(i.center.x, i.center.y, 0))
        self.observed_features = obstacleList

    def state_prediction(self, state_past, cov_past, U_t):
        # motion input in robot frame
        d_x = U_t.x * self.d_t
        d_y = U_t.y * self.d_t
        d_theta = U_t.theta * self.d_t
        theta = state_past.theta
        theta_ = state_past.theta + d_theta/2
        s_theta = math.sin(theta)
        c_theta = math.cos(theta)
        s__theta = math.sin(theta_)
        c__theta = math.cos(theta_)
        
        # Jacobian matrix for Ekf linearization
        Gt = np.mat([[1, 0, -d_x * s_theta - d_y * c_theta],
                     [0, 1, d_x * c_theta - d_y * s_theta],
                     [0, 0, 1]])
        
        Wt = np.mat([[c__theta, -s__theta, -d_x * s__theta/2 - d_y * c__theta/2],
                     [s__theta, c__theta, d_x * c__theta/2 - d_y * s__theta/2],
                     [0, 0, 1]])

        # Calculate model covariance
        stdev_vec = self.Model_const * np.mat([[d_x], [d_y], [d_theta]])
        
        var_x = stdev_vec[0,0] * stdev_vec[0,0]
        var_y = stdev_vec[1,0] * stdev_vec[1,0]
        var_theta = stdev_vec[2,0] * stdev_vec[2,0]
        
        cov_motion = np.mat([[var_x, 0, 0],
                             [0, var_y, 0],
                             [0, 0, var_theta]])
        
        # Prediction Mean
        x_pre = state_past.x + d_x * c__theta - d_y * s__theta
        y_pre = state_past.y + d_x * s__theta + d_y * c__theta
        theta_pre = state_past.theta + d_theta
        state_pre = state_vector(x_pre, y_pre, theta_pre)

        # Covariance of Prediction
        Gt_T = np.transpose(Gt)
        Wt_T = np.transpose(Wt)
        cov_pre = Gt * cov_past * Gt_T + Wt * cov_motion * Wt_T
        # print(cov_pre)
        # self.state_measurement_update(state_pre, cov_pre, self.observed_features)
        # rviz visualization
        self.prediction_publish(state_pre, cov_pre)
        return [state_pre, cov_pre]

    def state_measurement_update(self, state_pre, cov_pre, measurementList):
        # Estimate correspondence and position of robot
        feature_list = []
        mean_est = state_pre
        cov_est = cov_pre
        for obs in self.observed_features:
            z_hat_list = []
            j_list = []
            H_list = []
            S_list = []
            j_max = 0
            # measurement data, observed feature in polar coordinates
            r_sense = self.distance(obs, state_vector(0,0,0))
            phi_sense = math.atan2(obs.y, obs.x)
            z = np.mat([[r_sense],
                        [phi_sense], 
                        [0]])
            print("------------------")
            print("observed features : ",[obs.x, obs.y])
            # print("observed features : ",[r_sense, phi_sense, self.observed_features.index(obs)])

            for lm in self.landmark_list:
                # distance from prediction pose to landmark pose
                q_sqrt = self.distance(lm, state_pre)
                q = q_sqrt * q_sqrt
                phi = self.theta_convert(math.atan2(lm.y - state_pre.y, lm.x - state_pre.x) - state_pre.theta)
                z_hat = np.mat([[q_sqrt], 
                                [phi_sense], 
                                [0]])
                z_hat_list.append(z_hat)

                H = np.mat([[-(lm.x - state_pre.x)/q_sqrt, -(lm.y - state_pre.y)/q_sqrt, 0],
                            [(lm.y - state_pre.y)/q, -(lm.x - state_pre.x)/q, -1],
                            [0, 0, 0]])
                H = self.matrix_tozero(H)
                H_list.append(H)

                H_t = np.transpose(H)

                cov_pre = self.matrix_tozero(cov_pre)
                S = H * cov_pre * H_t + self.Q
                S = self.matrix_tozero(S)
                S_list.append(S)

                d_z = z - z_hat
                d_z[1,0] = self.theta_convert(d_z[1,0])
                d_z_T = np.transpose(d_z)
                S_inv = np.linalg.inv(S)
                det_2piS = np.linalg.det(2 * math.pi * S)


                if det_2piS != 0:
                    try:
                        exp = math.exp(-0.5 * d_z_T * S_inv * d_z)
                    except OverflowError:
                        exp = float('inf')
                    j =  pow(det_2piS, -0.5) * math.exp(-0.5 * d_z_T * S_inv * d_z)

                    # j =  math.exp(-0.5 * d_z_T * S_inv * d_z)
                    # j = exp(-0.5*transpose(z - z_hat)*inv(S)*(z - z_hat))* (det(2*pi*S))^(-0.5)
                else:
                    j = 0
                j_list.append(j)

                print("z_hat = \n", z_hat, end="\n\n")
                print("H = \n", H, end="\n\n")
                print("Cov_pre = \n", cov_pre, end="\n\n")
                print("H_t = \n", H_t, end="\n\n")
                print("S = \n", S, end="\n\n")
                print("2piS = \n", 2 * math.pi * S, end="\n\n")
                print("det 2piS = \n", det_2piS, end="\n\n")
                print("landmark", [lm.x, lm.y, lm.theta], ", j = ", j)

            j_max = max(j_list)
            print("j_max = ", j_max)
            idx = j_list.index(j_max)
            H = H_list[idx]
            S = S_list[idx]
            z_hat = z_hat_list[idx]
            # Feat = update_feature(j_max, H, S, z, z_hat, self.measurement_update_threshold)
            # feature_list.append(Feat)

            if j_max > self.measurement_update_threshold:
                K = cov_pre * H_t * S_inv
                delta_z = z - z_hat
                # ensure delta_z phi in the domain of [-pi, pi]
                delta_phi = self.theta_convert(delta_z[1,0])
                delta_r = delta_z[0,0]
                delta_z = np.mat([[delta_r],
                                [delta_phi], 
                                [0]])
                mean_est_matrix = state_pre.matrix() + K * delta_z
                mean_est_theta = self.theta_convert(mean_est_matrix[2,0])
                mean_est = state_vector(mean_est_matrix[0,0], 
                                        mean_est_matrix[1,0],
                                        mean_est_theta)
                I = (3,3)
                I = np.ones(I)
                cov_est = (I - K * H) * cov_pre
                print("updated")
            else:
                mean_est = state_pre
                cov_est = cov_pre
                print("fuckk")

        self.estimation_publish(mean_est, cov_est)
        # self.transform_publish(mean_est, self.odom_bf)
        # self.State_past = mean_est
        # self.Cov_past = cov_est
        # rospy.Rate(30).sleep()
        print("============================")

    def transform_publish(self, map_bf, odom_bf):
        map_odom = state_vector(0,0,0)
        map_odom.x = map_bf.x - odom_bf.x
        map_odom.y = map_bf.y - odom_bf.y
        map_odom.theta = self.theta_convert(map_bf.theta - odom_bf.theta)
        rate = rospy.Rate(30)
        quat = quaternion_from_euler(0,0,map_odom.theta)
        pose = (map_odom.x, map_odom.y, 0)
        self.tf_broadcaster.sendTransform(pose, quat, rospy.Time.now(), "odom", "map")
        rate.sleep()

    def prediction_publish(self, state, cov_mat):
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = rospy.get_rostime()
        pose_msg.header.frame_id = "odom"
        pose_msg.pose.pose.position.x = state.x
        pose_msg.pose.pose.position.y = state.y
        quaternion = quaternion_from_euler(0, 0, state.theta)
        pose_msg.pose.pose.orientation.x = quaternion[0]
        pose_msg.pose.pose.orientation.y = quaternion[1]
        pose_msg.pose.pose.orientation.z = quaternion[2]
        pose_msg.pose.pose.orientation.w = quaternion[3]

        pose_msg.pose.covariance = [cov_mat[0,0], 0, 0, 0, 0, 0,
                                    0, cov_mat[1,1], 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, 0,
                                    0, 0, 0, 0, 0, cov_mat[2,2]]

        self.prediction_pub.publish(pose_msg)

    def estimation_publish(self, state, cov_mat):
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = rospy.get_rostime()
        pose_msg.header.frame_id = "map"
        pose_msg.pose.pose.position.x = state.x
        pose_msg.pose.pose.position.y = state.y
        quaternion = quaternion_from_euler(0, 0, state.theta)

        pose_msg.pose.pose.orientation.x = quaternion[0]
        pose_msg.pose.pose.orientation.y = quaternion[1]
        pose_msg.pose.pose.orientation.z = quaternion[2]
        pose_msg.pose.pose.orientation.w = quaternion[3]

        pose_msg.pose.covariance = [cov_mat[0,0], 0, 0, 0, 0, 0,
                               0, cov_mat[1,1], 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, 0,
                               0, 0, 0, 0, 0, cov_mat[2,2]]
        self.estimation_pub.publish(pose_msg)

    def distance(self, p1, p2):
        d_x = p1.x - p2.x
        d_y = p1.y - p2.y
        return math.sqrt(d_x * d_x + d_y * d_y)
    
    def theta_convert(self, input):
        # convert rad domain to [-pi, pi]
        pi = math.pi
        if input >=0:
            input = math.fmod(input, 2*pi)
            if input > pi:
                input -= 2*pi
                output = input
            else:
                output = input
        else:
            input *= -1
            input = math.fmod(input, 2*pi)
            if input > pi:
                input -= 2*pi
                output = input*-1
            else:
                output = input*-1
        return output

    def theta_error_signed(self, theta_start, theta_final):
        curPos_vx = math.cos(theta_start)
        curPos_vy = math.sin(theta_start)
        goalPos_vx = math.cos(theta_final)
        goalPos_vy = math.sin(theta_final)
        dot = round(curPos_vx * goalPos_vx + curPos_vy * goalPos_vy, 5)
        theta_err = math.acos(dot)

        if abs(self.theta_convert(theta_start + theta_err) - theta_final) > 0.001:
            theta_err *= -1
        return theta_err

    def matrix_tozero(self, matrix):
        rows = matrix.shape[0]
        columns = matrix.shape[1]
        for i in range(rows):
            for j in range(columns):
                if abs(matrix[i,j]) < 0.00001:
                    matrix[i,j] = 0
        return matrix

if __name__ == '__main__':
    rospy.init_node('ekf_localization', anonymous = True)
    loc = localization()
    rospy.spin()
    # loc.start()