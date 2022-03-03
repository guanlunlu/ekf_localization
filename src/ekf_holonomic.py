#!/usr/bin/python3
import numpy as np
import math

# for ros
import rospy
import operator
from obstacle_detector.msg import Obstacles
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
from tf import TransformBroadcaster
import tf.transformations as tf_t
from obstacle_detector.msg import Obstacles
from obstacle_detector.msg import CircleObstacle
from sensor_msgs.msg import Imu

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

    def distanceto(self, p):
        return math.sqrt(pow(self.x-p.x, 2) + pow(self.y-p.y, 2))

class update_feature:
    def __init__(self, j, H, S, z, z_hat, landmark, landmark_scan):
        self.j = j
        self.H = H
        self.H_t = np.transpose(H)
        self.S = S
        self.z = z
        # feature measument vector
        self.z_hat = z_hat
        # the landmark which feature corresponded to ([[x],[y]])
        self.landmark = landmark
        # raw feature scanned by LiDAR (basefootprint origin)([[x],[y]])
        self.landmark_scan = landmark_scan
        self.mini_likelihood = 0.3
        # feature heuristic function
        self.score = 0
        self.likelihood_weight = 1.1
        self.translation_const= 0.3
        self.trans_sat = 5
        self.rotation_const = 0.5
        self.rot_sat = 5
        self.score_threshold = 12
        # self.score_threshold = 15
    
    def update_condition(self, vec_err):
        trans_err = abs(vec_err[0,0])
        rot_err = abs(self.theta_convert(vec_err[1,0]))
        # print("err", (trans_err, rot_err))
        if round(trans_err,10) != 0:
            trans_score = self.translation_const/trans_err
        else:
            trans_score = self.trans_sat

        if round(rot_err, 10) != 0:
            rot_score = self.rotation_const/rot_err
        else:
            rot_score = self.rot_sat
        self.score = trans_score + rot_score + self.j * self.likelihood_weight
        print(self.score)
        if self.score > self.score_threshold:
            return 1
        else:
            return 0

    def update(self, state_pre, cov_pre):
        if self.j > self.mini_likelihood and self.j != np.nan:
        # if self.update_condition(self.z-self.z_hat):
            K_i = cov_pre@self.H.T@np.linalg.inv(self.S)
            d_z = self.z-self.z_hat
            d_z[1,0] = self.theta_convert(d_z[1,0])
            mu_bar = state_pre + K_i@(d_z)
            sigma_bar = (np.eye(3) - self._fix_FP_issue(K_i@self.H))@cov_pre
            return [mu_bar, sigma_bar]
        else:
            return "update error"

    def _fix_FP_issue(self, matrix, upper_bound=1e-10):
        matrix[matrix<upper_bound] = 0
        return matrix

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
class EKF:
    def __init__(self):
        self.landmark1 = state_vector(1.0, -0.05, 0)
        self.landmark2 = state_vector(1.95, 3.1, 0)
        self.landmark3 = state_vector(0.05, 3.1, 0)
        # for beacon position ([x], [y])
        self.beacon_position = np.array([[self.landmark1.x, self.landmark2.x, self.landmark3.x],\
                                         [self.landmark1.y, self.landmark2.y, self.landmark3.y]])


        # for robot initial state (x; y; phi(-pi,pi))
        self.mu_0 = np.array([[0.5],\
                              [0.5],\
                              [0.5*np.pi]])

        # self.mu_0 = np.array([[0.55],\
        #                       [2.65],\
        #                       [-0.5*np.pi]])
        
        # for robot state
        self.mu_past = self.mu_0.copy()
        self.mu = np.array([[0.0],\
                            [0.0],\
                            [0.0]])

        self.Model_const = np.mat([[6, 0, 0],
                                   [0, 6, 0],
                                   [0, 0, .8]])

        self.sigma_past = np.zeros((3,3))
        self.sigma = np.zeros((3,3))

        self.stamp_past = rospy.get_time()

        # for beacon pillar detection
        self.if_new_obstacles = False
        self.beacon_scan = np.nan
        
        # imu measurement
        self.imu_w = 0
        # geometric validation param.
        self.sss_equiv_threshold = 0.07
        # for ros
        self.odom_sub = rospy.Subscriber("odom", Odometry, self.odom_sub_callback)
        # self.obstacles_sub = rospy.Subscriber("raw_obstacles", Obstacles, self.obstacles_sub_callback)
        self.obstacles_sub = rospy.Subscriber("landmark_extracted", Obstacles, self.obstacles_sub_callback)
        # self.imu_sub = rospy.Subscriber("imu", Imu, self.imu_sub_callback)
        # self.initialpose_sub = rospy.Subscriber("initialpose", Obstacles, self.initialpose_sub_callback)
        self.ekf_pose_pub = rospy.Publisher("ekf_pose", PoseWithCovarianceStamped, queue_size=10)
        self.landmark_rviz = rospy.Publisher('landmark_updated', Obstacles, queue_size=10)
        self.ekf_pose_tfbr = TransformBroadcaster()
        
        self.map_frame_name = "map"
        self.robot_frame_name = "base_footprint"
        self.updated_landmark_scan = []
        
    def ekf_localization(self, v_x, v_y, w):
        # holonomic drive state prediction
        U_t = state_vector(v_x, v_y, w)
        mu_p = self.mu_past.copy()
        state_pre = state_vector(mu_p[0,0], mu_p[1,0], mu_p[2,0])
        mu_bar, sigma_bar = self.state_prediction(state_pre, self.sigma_past, U_t)
        sigma_bar = self._fix_FP_issue(sigma_bar)
        mu_bar = np.array([[mu_bar.x],\
                          [mu_bar.y],\
                          [mu_bar.theta]])

        # set a minimum likelihood value
        mini_likelihood = 0.4
        # ekf update step:
        # Q = np.diag((0.02, 0.02, 0.05))
        Q = np.diag((0.01, 0.01, 0.02))

        if self.if_new_obstacles is True:
            self.updated_landmark_scan = []
            L1_feat_list = []
            L2_feat_list = []
            L3_feat_list = []
            # for every obstacle (or beacon pillar), check the distance between real beacon position and itself.
            for landmark_scan in np.nditer(self.beacon_scan, flags=['external_loop'], order='F'):
                # transfer landmark_scan type from (x, y) to (r, phi)
                landmark_scan = np.reshape(landmark_scan, (2,1))
                z_i = self._cartesian_to_polar(landmark_scan, np.zeros((3,1)))
                j_max = -10000000
                d_min = 100000000
                H_j_max = 0
                S_j_max = 0
                z_j_max = 0

                landmark_kmax_x = -10
                landmark_kmax_y = -10
                for k in range(self.beacon_position.shape[1]):
                    landmark_k = np.reshape(self.beacon_position[:,k], (2,1))
                    z_k, H_k = self._cartesian_to_polar(landmark_k, mu_bar, cal_H=True)
                    landmark_k_cart = np.array([[z_k[0,0] * np.cos(z_k[1,0])], [z_k[0,0] * np.sin(z_k[1,0])]])
                    S_k = H_k@sigma_bar@H_k.T + Q
                    S_k = self._fix_FP_issue(S_k)
                    try:
                        # original 
                        # j_k = 1/np.sqrt(np.linalg.det(2*np.pi*S_k)) * np.exp(-0.5*(z_i-z_k).T@np.linalg.inv(S_k)@(z_i-z_k))
                        d = self._euclidean_distance(landmark_scan, landmark_k_cart)
                        
                        # ln(j_k()) version
                        d_z = z_i-z_k
                        d_z[1,0] = self.theta_convert(d_z[1,0])
                        # print(d_z)
                        j_k = -0.5*(d_z).T@np.linalg.inv(S_k)@(d_z) - np.log(np.sqrt(np.linalg.det(2*np.pi*S_k)))
                        # print(d)
                        if np.around(j_k, 10)[0,0] > np.around(j_max, 10):
                        # if d < d_min:
                            d_min = d
                            landmark_kmax_x = landmark_k[0,0]
                            landmark_kmax_y = landmark_k[1,0]
                            j_max = j_k.copy()
                            H_j_max = H_k.copy()
                            S_j_max = S_k.copy()
                            z_j_max = z_k.copy()
                    except Exception as e:
                        rospy.logerr("%s", e)
                        
                # rospy.loginfo("j_max = %s", j_max)
                # rospy.loginfo_throttle(0.5, "j_max = %s", j_max)
                feat = update_feature(j_max, H_j_max, S_j_max, z_i, z_j_max, landmark_k, landmark_scan)
                if landmark_kmax_x == self.landmark1.x and landmark_kmax_y == self.landmark1.y:
                    L1_feat_list.append(feat)
                if landmark_kmax_x == self.landmark2.x and landmark_kmax_y == self.landmark2.y:
                    L2_feat_list.append(feat)
                if landmark_kmax_x == self.landmark3.x and landmark_kmax_y == self.landmark3.y:
                    L3_feat_list.append(feat)

            if len(L1_feat_list) != 0:
                L1_feat = max(L1_feat_list, key=operator.attrgetter('j'))
                if L1_feat.update(mu_bar, sigma_bar) != "update error":
                    mu_bar, sigma_bar = L1_feat.update(mu_bar, sigma_bar)
                    lx = L1_feat.landmark_scan[0,0]
                    ly = L1_feat.landmark_scan[1,0]
                    self.updated_landmark_scan.append(state_vector(lx, ly, 0))

            if len(L2_feat_list) != 0:
                L2_feat = max(L2_feat_list, key=operator.attrgetter('j'))
                if L2_feat.update(mu_bar, sigma_bar) != "update error":
                    mu_bar, sigma_bar = L2_feat.update(mu_bar, sigma_bar)
                    lx = L2_feat.landmark_scan[0,0]
                    ly = L2_feat.landmark_scan[1,0]
                    self.updated_landmark_scan.append(state_vector(lx, ly, 0))

            if len(L3_feat_list) != 0:
                L3_feat = max(L3_feat_list, key=operator.attrgetter('j'))
                if L3_feat.update(mu_bar, sigma_bar) != "update error":
                    mu_bar, sigma_bar = L3_feat.update(mu_bar, sigma_bar)
                    lx = L3_feat.landmark_scan[0,0]
                    ly = L3_feat.landmark_scan[1,0]
                    self.updated_landmark_scan.append(state_vector(lx, ly, 0))
        # rospy.loginfo_throttle(0.5, "------------------")
        # print("--------")


        self.mu = mu_bar.copy()
        self.sigma = self._fix_FP_issue(sigma_bar.copy())
        self.mu_past = self.mu.copy()
        self.sigma_past = self.sigma.copy()
        # finish once ekf, change the flag
        self.if_new_obstacles = False
        # publish updated lanmark data to rviz
        self.show_landmark_extracted(self.updated_landmark_scan)

    def geometric_validation(self,mpost1, mpost2, mpost3):
        mdis23 = mpost2.distanceto(mpost3)
        mdis12 = mpost1.distanceto(mpost2)
        mdis13 = mpost1.distanceto(mpost3)
        # SSS equivalent condition
        if abs(mdis12 - self.landmark1.distanceto(self.landmark2)) < self.sss_equiv_threshold and abs(mdis13 - self.landmark1.distanceto(self.landmark3)) < self.sss_equiv_threshold and abs(mdis23 - self.landmark2.distanceto(self.landmark3)) < self.sss_equiv_threshold:
            return 1
        else:
            return 0

    def state_prediction(self, state_past, cov_past, U_t):
        # motion input in robot frame
        # d_x = U_t.x * self.d_t
        # d_y = U_t.y * self.d_t
        # d_theta = U_t.theta * self.d_t
        d_x = U_t.x /100
        d_y = U_t.y /100
        d_theta = U_t.theta /100
        theta = state_past.theta
        theta_ = state_past.theta + d_theta/2
        s_theta = np.sin(theta)
        c_theta = np.cos(theta)
        s__theta = np.sin(theta_)
        c__theta = np.cos(theta_)
        
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
        return [state_pre, cov_pre]

    def _euclidean_distance(self, a, b):
        return np.sqrt((b[1, 0]-a[1, 0])**2 + (b[0, 0]-a[0, 0])**2)

    def _angle_limit_checking(self, theta):
        if theta > np.pi:
            theta -= 2 * np.pi
        elif theta <= -np.pi:
            theta += 2 * np.pi
        return theta

    # find polar coordinate for point a from ref point b
    def _cartesian_to_polar(self, a, b, cal_H=False):
        q_sqrt = self._euclidean_distance(a, b)
        q = q_sqrt**2
        a_b_x = a[0, 0]-b[0, 0]
        a_b_y = a[1, 0]-b[1, 0]
        z_hat = np.array([[q_sqrt],\
                          [np.arctan2(a_b_y, a_b_x) - b[2, 0]],\
                          [1]])
        z_hat[1,0] = self._angle_limit_checking(z_hat[1,0])
        if cal_H:
            H = np.array([[-(a_b_x/q_sqrt), -(a_b_y/q_sqrt),  0],\
                          [        a_b_y/q,      -(a_b_x/q), -1],\
                          [              0,               0,  0]])
            return (z_hat, H)
        else:
            return z_hat

    # rotate theta and translation (x, y) from the laser frame 
    def _tf_laser_to_map(self, mu_bar, landmark_scan):
        s = np.sin(mu_bar[2,0])
        c = np.cos(mu_bar[2,0])
        landmark_scan = np.array([[c, -s],[s, c]])@landmark_scan + np.reshape(mu_bar[0:2, 0], (2,1))
        return landmark_scan

    # if value less than upper_bound then equal to 0
    def _fix_FP_issue(self, matrix, upper_bound=1e-10):
        matrix[matrix<upper_bound] = 0
        return matrix

    def odom_sub_callback(self, odom):
        stamp = rospy.get_time()
        time_stamp = rospy.Time.now()
        v_x = odom.twist.twist.linear.x
        v_y = odom.twist.twist.linear.y
        w = odom.twist.twist.angular.z
        # w = self.imu_w
        self.ekf_localization(v_x, v_y, w)
        self.publish_ekf_pose(time_stamp + rospy.Duration(0.2))
        self.broadcast_ekf_pos_tf(odom)
        # rospy.loginfo("ekf time: %s", rospy.get_time()-stamp)

    def obstacles_sub_callback(self, obstacles):
        self.if_new_obstacles = False
        self.beacon_scan = np.nan
        for item in obstacles.circles:
            center_xy = np.array([[item.center.x],[item.center.y]])
            if self.beacon_scan is np.nan:
                self.beacon_scan = center_xy
            else:
                self.beacon_scan = np.hstack([self.beacon_scan, center_xy])
        self.if_new_obstacles = True

    def publish_ekf_pose(self, stamp):
        pose = PoseWithCovarianceStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = self.map_frame_name
        pose.pose.pose.position.x = self.mu[0,0]
        pose.pose.pose.position.y = self.mu[1,0]
        q = tf_t.quaternion_from_euler(0, 0, self.mu[2,0])
        pose.pose.pose.orientation.x = q[0]
        pose.pose.pose.orientation.y = q[1]
        pose.pose.pose.orientation.z = q[2]
        pose.pose.pose.orientation.w = q[3]
        pose.pose.covariance[0] = self.sigma[0,0] # x-x
        pose.pose.covariance[1] = self.sigma[0,1] # x-y
        pose.pose.covariance[5] = self.sigma[0,2] # x-theta
        pose.pose.covariance[6] = self.sigma[1,0] # y-x
        pose.pose.covariance[7] = self.sigma[1,1] # y-y
        pose.pose.covariance[11] = self.sigma[1,2] # y-theta
        pose.pose.covariance[30] = self.sigma[2,0] # theta-x
        pose.pose.covariance[31] = self.sigma[2,1] # theta-y
        pose.pose.covariance[35] = self.sigma[2,2] # theta-theta
        self.ekf_pose_pub.publish(pose)

    def broadcast_ekf_pos_tf(self, odom):
        map_to_baseft = tf_t.concatenate_matrices(
                        tf_t.translation_matrix((self.mu[0,0], self.mu[1,0], 0)),
                        tf_t.quaternion_matrix(tf_t.quaternion_from_euler(0, 0, self.mu[2,0])))
        # baseft_to_map = tf_t.inverse_matrix(map_to_baseft)
        odom_to_baseft = tf_t.concatenate_matrices(
                         tf_t.translation_matrix((odom.pose.pose.position.x, odom.pose.pose.position.y, 0)),
                         tf_t.quaternion_matrix((0, 0, odom.pose.pose.orientation.z, odom.pose.pose.orientation.w)))
        baseft_to_odom =  tf_t.inverse_matrix(odom_to_baseft)
        map_to_odom = tf_t.concatenate_matrices(map_to_baseft, baseft_to_odom)

        self.ekf_pose_tfbr.sendTransform(tf_t.translation_from_matrix(map_to_odom),
                                         tf_t.quaternion_from_matrix(map_to_odom),
                                         odom.header.stamp,
                                         odom.header.frame_id,
                                         self.map_frame_name)

    def show_landmark_extracted(self, landmarks):
        msg = Obstacles()
        msg.header.stamp = rospy.Time(0)
        msg.header.frame_id = "base_footprint"
        circleList = []
        
        for i in landmarks:
            circle = CircleObstacle()
            circle.center.x = i.x
            circle.center.y = i.y
            circle.radius = 0.15
            circle.true_radius = 0.1
            circleList.append(circle)
        msg.circles = circleList
        self.landmark_rviz.publish(msg)

    def initialpose_sub_callback(self, data):
        x = data.pose.pose.position.x
        y = data.pose.pose.position.y
        qx = data.pose.pose.orientation.x
        qy = data.pose.pose.orientation.y
        qz = data.pose.pose.orientation.z
        qw = data.pose.pose.orientation.w
        roll, pitch, yaw = tf_t.euler_from_quaternion([qx, qy, qz, qw])
        
        self.mu_past = np.array([[x],\
                                 [y],\
                                 [yaw]])

    def imu_sub_callback(self, imu):
        self.imu_w = imu.angular_velocity.z

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

    def theta_error(self,theta1, theta2):
        curPos_vx = math.cos(theta1)
        curPos_vy = math.sin(theta1)
        goalPos_vx = math.cos(theta2)
        goalPos_vy = math.sin(theta2)
        theta_err = math.acos(curPos_vx * goalPos_vx + curPos_vy * goalPos_vy)
        return theta_err

if __name__ == '__main__':
    rospy.init_node('ekf_localization', anonymous=True)
    ekf = EKF()
    while not rospy.is_shutdown():
        rospy.spin()