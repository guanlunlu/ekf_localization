#!/usr/bin/env python3
import rospy
import math
import numpy as np
import itertools
import tf
from tf.transformations import quaternion_from_euler
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Point
from geometry_msgs.msg import PoseWithCovarianceStamped
from obstacle_detector.msg import Obstacles
from obstacle_detector.msg import CircleObstacle
from std_msgs.msg import String
pi = math.pi

class pose():
    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta
    
    def distanceto(self, p):
        return math.sqrt(pow(self.x-p.x, 2) + pow(self.y-p.y, 2))
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
        pass

class localization():
    def __init__(self):
        self.tf_listener = tf.TransformListener()
        self.obstacleSub = rospy.Subscriber("/raw_obstacles", Obstacles, self.obstacleCallback)
        self.landmark_rviz = rospy.Publisher('landmark_extracted', Obstacles, queue_size=10)
        self.landmark1 = pose(0.05, 3.1, 0)
        self.landmark2 = pose(1.0, -0.05, 0)
        self.landmark3 = pose(1.95, 3.1, 0)
        self.landmarkList = [self.landmark1, self.landmark2, self.landmark3]
        self.landmark_extraction_threshold = 1
        # self.landmark_extraction_threshold = 1 # 0.6
        # self.field_mode = "wall"

    def obstacleCallback(self, data):
        obstacleList = []
        for i in data.circles:
            # rospy.loginfo("r", i.radius)
            obstacleList.append(pose(i.center.x, i.center.y, 0))
    
        extractedList = self.landmark_extraction(obstacleList)
        if extractedList != "Not found":
            self.landmark_extracted_publish(extractedList)
        else:
            print("Landmark Not Found !")

    def landmark_extraction(self, raw_obsList):
        tf1_success = 0
        tf2_success = 0
        tf3_success = 0
        if self.tf_listener.canTransform('base_footprint', 'landmark1', rospy.Time(0)):
            bf_landmark1 = self.tf_listener.lookupTransform('base_footprint', 'landmark1', rospy.Time(0))
            bf_landmark1 = pose(bf_landmark1[0][0], bf_landmark1[0][1], 0)
            tf1_success = 1
        if self.tf_listener.canTransform('base_footprint', 'landmark2', rospy.Time(0)):
            bf_landmark2 = self.tf_listener.lookupTransform('base_footprint', 'landmark2', rospy.Time(0))
            bf_landmark2 = pose(bf_landmark2[0][0], bf_landmark2[0][1], 0)
            tf2_success = 1
        if self.tf_listener.canTransform('base_footprint', 'landmark3', rospy.Time(0)):
            bf_landmark3 = self.tf_listener.lookupTransform('base_footprint', 'landmark3', rospy.Time(0))
            bf_landmark3 = pose(bf_landmark3[0][0], bf_landmark3[0][1], 0)
            tf3_success = 1
        if tf1_success and tf2_success and tf3_success:
            min_err1 = 10000000000
            min_err2 = 10000000000
            min_err3 = 10000000000
            lm1_extracted = pose(0,0,0)
            lm2_extracted = pose(0,0,0)
            lm3_extracted = pose(0,0,0)
            lm1_captured = 0
            lm2_captured = 0
            lm3_captured = 0
            lm_list = []
            for obs in raw_obsList:
                err1 = obs.distanceto(bf_landmark1)
                err2 = obs.distanceto(bf_landmark2)
                err3 = obs.distanceto(bf_landmark3)
                if err1 < self.landmark_extraction_threshold:
                    if err1 < min_err1:
                        min_err1 = err1
                        lm1_extracted = obs
                        lm1_captured = 1
                if err2 < self.landmark_extraction_threshold:
                    if err2 < min_err2:
                        min_err2 = err2
                        lm2_extracted = obs
                        lm2_captured = 1
                if err3 < self.landmark_extraction_threshold:
                    if err3 < min_err3:
                        min_err3 = err3
                        lm3_extracted = obs
                        lm3_captured = 1
            if lm1_captured:
                lm_list.append(lm1_extracted)
            if lm2_captured:
                lm_list.append(lm2_extracted)
            if lm3_captured:
                lm_list.append(lm3_extracted)
            return lm_list
        else:
            return "Not found"

    def landmark_extracted_publish(self, landmarks):
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
        
if __name__ == '__main__':
    try:
        rospy.init_node('landmark_extractor', anonymous = True)
        getPose = localization()
        # getPose.init()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass