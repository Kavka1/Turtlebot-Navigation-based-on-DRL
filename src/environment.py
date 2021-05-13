import os
import rospy
import numpy as np
import math
import random

from geometry_msgs.msg import Twist, Point, Pose
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty
from gazebo_msgs.srv import SpawnModel, DeleteModel, SetModelState, SetModelStateRequest
from gazebo_msgs.msg import ModelState

diagonal_dis = math.sqrt(2) * (3.6 + 3.8)
goal_model_dir = os.path.join(os.path.split(os.path.realpath(__file__))[0], '..', '..', '..', '..', 'model_editor_models', 'Target_', 'model.sdf')

class Env():
    def __init__(self, args):
        self.position = Pose()
        self.goal_position = Pose()
        self.goal_position.position.x = 0
        self.goal_position.position.y = 0
        self.pub_cmd_vel = rospy.Publisher('cmd_vel', Twist, queue_size=10)
        self.sub_odom = rospy.Subscriber('odom', Odometry, self.getOdometry)
        self.reset_proxy = rospy.ServiceProxy('gazebo/reset_simulation', Empty)
        self.unpause_proxy = rospy.ServiceProxy('gazebo/unpause_physics', Empty)
        self.pause_proxy = rospy.ServiceProxy('gazebo/pause_physics', Empty)
        self.goal = rospy.ServiceProxy('/gazebo/spawn_sdf_model', SpawnModel)
        self.del_model = rospy.ServiceProxy('/gazebo/delete_model', DeleteModel)
        self.turtle_reset = rospy.ServiceProxy('gazebo/set_model_state', SetModelState)
        self.s_dim = 14
        self.a_dim = 2
        self.past_distance = 0.
        self.threshold_arrive = 0.25
        self.min_range = 0.15
        self.maze_bound = 2.5#1.5 if args.is_training else 2.5

        #Spawn the target
        rospy.wait_for_service('/gazebo/spawn_sdf_model')
        try:
            goal_urdf = open(goal_model_dir, "r").read()
            target = SpawnModel
            target.model_name = 'target'  # the same with sdf name
            target.model_xml = goal_urdf
            self.goal(target.model_name, target.model_xml, 'namespace', self.goal_position, 'world')
        except (rospy.ServiceException) as e:
            print("/gazebo/failed to build the target")
        rospy.wait_for_service('/gazebo/unpause_physics')

    def getGoalDistace(self):
        goal_distance = math.hypot(self.goal_position.position.x - self.position.x, self.goal_position.position.y - self.position.y)
        self.past_distance = goal_distance
        return goal_distance

    def getOdometry(self, odom):
        self.position = odom.pose.pose.position
        orientation = odom.pose.pose.orientation
        q_x, q_y, q_z, q_w = orientation.x, orientation.y, orientation.z, orientation.w
        yaw = round(math.degrees(math.atan2(2 * (q_x * q_y + q_w * q_z), 1 - 2 * (q_y * q_y + q_z * q_z))))

        if yaw >= 0:
             yaw = yaw
        else:
             yaw = yaw + 360

        rel_dis_x = round(self.goal_position.position.x - self.position.x, 1)
        rel_dis_y = round(self.goal_position.position.y - self.position.y, 1)

        # Calculate the angle between robot and target
        if rel_dis_x > 0 and rel_dis_y > 0:
            theta = math.atan(rel_dis_y / rel_dis_x)
        elif rel_dis_x > 0 and rel_dis_y < 0:
            theta = 2 * math.pi + math.atan(rel_dis_y / rel_dis_x)
        elif rel_dis_x < 0 and rel_dis_y < 0:
            theta = math.pi + math.atan(rel_dis_y / rel_dis_x)
        elif rel_dis_x < 0 and rel_dis_y > 0:
            theta = math.pi + math.atan(rel_dis_y / rel_dis_x)
        elif rel_dis_x == 0 and rel_dis_y > 0:
            theta = 1 / 2 * math.pi
        elif rel_dis_x == 0 and rel_dis_y < 0:
            theta = 3 / 2 * math.pi
        elif rel_dis_y == 0 and rel_dis_x > 0:
            theta = 0
        else:
            theta = math.pi
        rel_theta = round(math.degrees(theta), 2)

        diff_angle = abs(rel_theta - yaw)

        if diff_angle <= 180:
            diff_angle = round(diff_angle, 2)
        else:
            diff_angle = round(360 - diff_angle, 2)

        self.rel_theta = rel_theta
        self.yaw = yaw
        self.diff_angle = diff_angle

    def getState(self, scan):
        scan_range = []
        current_distance = math.hypot(self.goal_position.position.x - self.position.x, self.goal_position.position.y - self.position.y)
        yaw = self.yaw
        rel_theta = self.rel_theta
        diff_angle = self.diff_angle
        die = 0
        arrive = 0
        for i in range(0, 359, 36):
            if scan.ranges[i] == float('Inf'):
                scan_range.append(3.5)
            elif np.isnan(scan.ranges[i]):
                scan_range.append(0)
            else:
                scan_range.append(scan.ranges[i])

        if self.min_range > min(scan_range) > 0:
            die = 1
        if current_distance <= self.threshold_arrive:
            arrive = 1

        return scan_range, current_distance, yaw, rel_theta, diff_angle, die, arrive

    def setReward(self, die, arrive, round_step):
        current_distance = math.hypot(self.goal_position.position.x - self.position.x, self.goal_position.position.y - self.position.y)
        distance_rate = (self.past_distance - current_distance)
        reward = (500 - round_step) * distance_rate * 0.001
        if die:
            reward = -100
        if arrive:
            reward = 100
        return reward

    def step(self, action, round_step):
        linear_vel = action[0]
        ang_vel = action[1]

        vel_cmd = Twist()
        vel_cmd.linear.x = linear_vel / 4
        vel_cmd.angular.z = ang_vel
        self.pub_cmd_vel.publish(vel_cmd)

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                print("no scan data")

        state, rel_dis, yaw, rel_theta, diff_angle, die, arrive = self.getState(data)
        state = [i / 3.5 for i in state]

        state = state + [rel_dis / diagonal_dis, yaw / 360, rel_theta / 360, diff_angle / 180]
        reward = self.setReward(die, arrive, round_step)

        return state, reward, die, arrive

    def reset(self):
        #reset the turtlebot
        objstate = SetModelStateRequest()
        objstate.model_state.model_name = 'turtlebot3_burger'
        objstate.model_state.pose.position.x = random.uniform(-self.maze_bound, self.maze_bound)
        objstate.model_state.pose.position.y = random.uniform(-self.maze_bound, self.maze_bound)
        self.turtle_reset(objstate)

        data = None
        while data is None:
            try:
                data = rospy.wait_for_message('scan', LaserScan, timeout=5)
            except:
                print("no scan data")

        self.goal_distance = self.getGoalDistace()
        state, rel_dis, yaw, rel_theta, diff_angle, done, arrive = self.getState(data)
        state = [i / 3.5 for i in state]

        state = state + [rel_dis / diagonal_dis, yaw / 360, rel_theta / 360, diff_angle / 180]

        return state
