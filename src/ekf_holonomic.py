#!/usr/bin/python3
import numpy as np

# for ros
import rospy
from obstacle_detector.msg import Obstacles
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
from tf import TransformBroadcaster
import tf.transformations as tf_t

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


class EKF:
    def __init__(self):
        # for beacon position ([x], [y])
        self.beacon_position = np.array([[0.05, 1.05, 1.95],\
                                         [ 3.1, -0.1,  3.1]])

        # for robot initial state (x; y; phi(-pi,pi))
        self.mu_0 = np.array([[0.5],\
                              [0.5],\
                              [0.5*np.pi]])
        
        # for robot state
        self.mu_past = self.mu_0.copy()
        self.mu = np.array([[0.0],\
                            [0.0],\
                            [0.0]])

        self.Model_const = np.mat([[0.4, 0, 0],
                                   [0, 0.4, 0],
                                   [0, 0, 0.4]])

        self.sigma_past = np.zeros((3,3))
        self.sigma = np.zeros((3,3))

        self.stamp_past = rospy.get_time()
        self.d_t = 0

        # for beacon pillar detection
        self.if_new_obstacles = False
        self.beacon_scan = np.nan

        # for ros
        self.odom_sub = rospy.Subscriber("odom", Odometry, self.odom_sub_callback)
        self.obstacles_sub = rospy.Subscriber("raw_obstacles", Obstacles, self.obstacles_sub_callback)
        self.ekf_pose_pub = rospy.Publisher("ekf_pose", PoseWithCovarianceStamped, queue_size=10)
        self.ekf_pose_tfbr = TransformBroadcaster()
        
        self.map_frame_name = "map"
        self.robot_frame_name = "base_footprint"
        
    def ekf_localization(self, v_x, v_y, w):
        # set motion covariance
        # a1 = 0.5
        # a2 = 0.8
        # a3 = 0.5
        # a4 = 0.8
        # # set a minimum likelihood value
        mini_likelihood = 0.5
        # # for convenience, ex: s_dt = sin(theta+wdt) 
        # theta = self.mu_past[2, 0].copy()
        # s = np.sin(theta)
        # c = np.cos(theta)
        # s_dt = np.sin(theta + w*self.dt)
        # c_dt = np.cos(theta + w*self.dt)
        
        # # ekf predict step:
        # if w < 1e-5 and w > -1e-5 :
        #     G = np.array([[1, 0, -v_x*s*self.dt],\
        #                   [0, 1,  v_x*c*self.dt],\
        #                   [0, 0,            1]])

        #     V = np.array([[c*self.dt, 0],\
        #                   [s*self.dt, 0],\
        #                   [0        , 0]])

        #     M = np.array([[a1*v**2 + a2*w**2,                 0],\
        #                   [                0, a3*v**2 + a4*w**2]])

        #     mu_bar = self.mu_past.copy() + np.array([[v*c*self.dt],\
        #                                              [v*s*self.dt],\
        #                                              [          0]])
        
        # else:
        #     G = np.array([[1, 0, (v*(-c+c_dt))/w],\
        #                   [0, 1, (v*(-s+s_dt))/w],\
        #                   [0, 0,               1]])

        #     V = np.array([[(-s+s_dt)/w,  v*(s-s_dt)/w**2 + v*self.dt*(c_dt)/w],\
        #                   [ (c-c_dt)/w, -v*(c-c_dt)/w**2 + v*self.dt*(s_dt)/w],\
        #                   [          0,                               self.dt]])

        #     M = np.array([[a1*v**2+a2*w**2,             0],\
        #                   [            0, a3*v**2+a4*w**2]])

        #     mu_bar = self.mu_past.copy() + np.array([[v*(-s+s_dt)/w],\
        #                                              [ v*(c-c_dt)/w],\
        #                                              [    w*self.dt]])
        #     mu_bar[2,0] = self._angle_limit_checking(mu_bar[2,0])

        # sigma_bar = G@self.sigma_past@G.T + V@M@V.T
        # sigma_bar = self._fix_FP_issue(sigma_bar)
        U_t = state_vector(v_x, v_y, w)
        mu_p = self.mu_past.copy()
        state_pre = state_vector(mu_p[0,0], mu_p[1,0], mu_p[2,0])
        mu_bar, sigma_bar = self.state_prediction(state_pre, self.sigma_past, U_t)
        sigma_bar = self._fix_FP_issue(sigma_bar)
        mu_bar = np.array([[mu_bar.x],\
                          [mu_bar.y],\
                          [mu_bar.theta]])

        # ekf update step:
        Q = np.diag((0.001, 0.2, 0.02))
        if self.if_new_obstacles is True:
            # for every obstacle (or beacon pillar), check the distance between real beacon position and itself.
            for landmark_scan in np.nditer(self.beacon_scan, flags=['external_loop'], order='F'):
                landmark_scan = np.reshape(landmark_scan, (2,1))
                # transfer landmark_scan type from (x, y) to (r, phi)
                z_i = self._cartesian_to_polar(landmark_scan, np.zeros((3,1)))
                j_max = 0
                H_j_max = 0
                S_j_max = 0
                z_j_max = 0
                for k in range(self.beacon_position.shape[1]):
                    landmark_k = np.reshape(self.beacon_position[:,k], (2,1))
                    z_k, H_k = self._cartesian_to_polar(landmark_k, mu_bar, cal_H=True)
                    S_k = H_k@sigma_bar@H_k.T + Q
                    S_k = self._fix_FP_issue(S_k)
                    try:
                        # original 
                        # j_k = 1/np.sqrt(np.linalg.det(2*np.pi*S_k)) * np.exp(-0.5*(z_i-z_k).T@np.linalg.inv(S_k)@(z_i-z_k))
                        
                        # ln(j_k()) version
                        j_k = -0.5*(z_i-z_k).T@np.linalg.inv(S_k)@(z_i-z_k) - np.log(np.sqrt(np.linalg.det(2*np.pi*S_k)))
                        if np.around(j_k, 10)[0,0] > np.around(j_max, 10):
                            j_max = j_k.copy()
                            H_j_max = H_k.copy()
                            S_j_max = S_k.copy()
                            z_j_max = z_k.copy()
                    except Exception as e:
                        rospy.logerr("%s", e)
                        # continue
                    # rospy.loginfo("H_k=%s, sigma_bar=%s, S_k=%s", H_k, sigma_bar, S_k)
                    # print("S_k-Q = ", H_k@sigma_bar@H_k.T, "H_k = ", H_k, "sigma_bar = ", sigma_bar)
                if j_max > mini_likelihood and j_max != np.nan:
                    K_i = sigma_bar@H_j_max.T@np.linalg.inv(S_j_max)
                    mu_bar = mu_bar + K_i@(z_i-z_j_max)
                    sigma_bar = (np.eye(3) - self._fix_FP_issue(K_i@H_j_max))@sigma_bar

        self.mu = mu_bar.copy()
        self.sigma = self._fix_FP_issue(sigma_bar.copy())
        self.mu_past = self.mu.copy()
        self.sigma_past = self.sigma.copy()
        # finish once ekf, change the flag
        self.if_new_obstacles = False

    def state_prediction(self, state_past, cov_past, U_t):
        # motion input in robot frame
        # d_x = U_t.x * self.d_t
        # d_y = U_t.y * self.d_t
        # d_theta = U_t.theta * self.d_t
        d_x = U_t.x /50
        d_y = U_t.y /50
        d_theta = U_t.theta /50
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
        # rospy.loginfo("odom time: %s, now: %s", odom.header.stamp, stamp)
        self.d_t = stamp - self.stamp_past
        v_x = odom.twist.twist.linear.x
        v_y = odom.twist.twist.linear.y
        w = odom.twist.twist.angular.z
        self.ekf_localization(v_x, v_y, w)
        self.publish_ekf_pose(time_stamp + rospy.Duration(0.2))
        self.broadcast_ekf_pos_tf(odom)

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

if __name__ == '__main__':
    rospy.init_node('ekf_localization', anonymous=True)
    ekf = EKF()
    while not rospy.is_shutdown():
        rospy.spin()