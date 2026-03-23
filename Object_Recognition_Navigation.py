'''

Warehouse-style pick and place robot for Stranger Things demo.
Navigates to two pickup locations, detects which character is present,
then delivers them to their assigned dropoff location.

Each pickup and dropoff is approached via a waypoint so the robot
always travels straight then turns, giving a consistent final orientation.

UPDATED: Pickups now use a 3-step approach:
  1. Waypoint (straight ahead, no turn)
  2. Pre-position (turn/orient here, safely away from the piece)
  3. Final approach (drive straight forward with DriveDistance — no spin)

'''

# start with Create3 docked, with the BACK of dock 0.5 m from (0,0), it will turn around when it undocks
# RUN THIS IN TERMINAL BEFORE BEGINNING: ros2 service call /reset_pose irobot_create_msgs/srv/ResetPose "{}"

#if needed (calibrate?): drive him to posiiton: ros2 run teleop_twist_keyboard teleop_twist_keyboard
#if needed (calibrate?): get position: ros2 topic echo /odom --once

import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from irobot_create_msgs.action import NavigateToPosition, Undock, DriveDistance
from keras.models import load_model
from picamera2 import Picamera2
from libcamera import controls
import cv2
import numpy as np
import time
from Gripper import gripper_close, gripper_open, gripper_cleanup

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# ==============================================================
# How far before the pickup the robot stops to do its spin,
# and then drives straight in to grab the piece.
# Increase PRE_POSITION_OFFSET if the gripper still clips.
# ==============================================================
PRE_POSITION_OFFSET = 0.12   # metres — robot turns here, away from the piece
FINAL_APPROACH_DIST = 0.12   # metres — straight drive in to the piece (should match offset)
FINAL_APPROACH_SPEED = 0.05  # m/s  — slow and gentle

# ==============================================================
# EDIT THESE ON DEMO DAY
#x — how far forward/backward. More negative = further forward. Less negative = not as far.
#y — how far left/right. More negative = further right. Less negative = not as far right.

# Assign each character to their dropoff location.
# Format: 'character_name': [(waypoint), (final position)]
# Waypoint: go straight to this x first, y=0, facing forward
# Final:    then turn and go to the actual dropoff position
# ==============================================================
DROPOFF_LOCATIONS = {
    'Eleven': [(-1.148, 0.0, 0.0, 1.0),   (-1.148, 0.724, 0.707, 0.707)],  # <-- swap with Will if needed
    'Will':   [(-2.172, 0.0, 0.0, 1.0),   (-2.172, 0.724, 0.707, 0.707)],  # <-- swap with Eleven if needed
}

# ==============================================================
# Each entry is [(waypoint), (final pickup position)]
# Waypoint: go straight to this x first, y=0, facing forward
# Final:    then turn and go to the actual pickup position
#
# The robot will stop PRE_POSITION_OFFSET metres short of the
# final y and spin there, then drive straight in.
# ==============================================================
PICKUP_LOCATIONS = [
    [(-1.229, 0.0, 0.0, 1.0),  (-1.229, -0.715, -0.707, 0.707)],   # Pickup station 1
    [(-2.13,  0.0, 0.0, 1.0),  (-2.13,  -0.715, -0.707, 0.707)],  # Pickup station 2
]


# Define the class StrangerThingsRobot as a subclass of Node
class StrangerThingsRobot(Node):

    # Define a method to initialize the node
    def __init__(self):

        # Initialize a node named stranger_things_robot
        super().__init__('stranger_things_robot')

        # Create an action client for undocking
        self._undock_client = ActionClient(self, Undock, 'undock')

        # Create an action client for navigation
        self._action_client = ActionClient(self, NavigateToPosition, 'navigate_to_position')

        # Create an action client for the straight final approach on pickups
        self._drive_client = ActionClient(self, DriveDistance, 'drive_distance')

        # Track which pickup we are currently on (0 or 1)
        self.current_pickup_index = 0

        # Track whether we are navigating to a pickup or a dropoff
        self.going_to_pickup = True

        # Navigation step tracker for each location:
        #   'waypoint'      — step 1: drive straight to the waypoint
        #   'pre_position'  — step 2 (pickups only): turn/orient here away from piece
        #   'final'         — step 3: either DriveDistance (pickup) or NavigateToPosition (dropoff)
        self.nav_step = 'waypoint'

        # Store the current dropoff coords so we can use them after the waypoint
        self.current_dropoff_coords = None

        # Load the Teachable Machine model and labels once at startup
        self.model = load_model('keras_model.h5', compile=False)
        self.class_names = open('labels.txt', 'r').readlines()
        self.get_logger().info('Model loaded successfully!')

        # Set up the picamera and leave it running for the whole session
        self.picam2 = Picamera2()
        self.picam2.set_controls({'AfMode': controls.AfModeEnum.Continuous})
        self.picam2.start()
        time.sleep(1)   # give the camera time to warm up
        self.get_logger().info('Camera ready!')

        self.get_logger().info('Stranger Things Robot initialized. Undocking...')

        # Kick off the sequence by undocking first
        self.undock()

    # ------------------------------------------------------------------
    # Undocking
    # ------------------------------------------------------------------

    def undock(self):
        goal_msg = Undock.Goal()
        self._undock_client.wait_for_server()
        self._send_goal_future = self._undock_client.send_goal_async(
            goal_msg, feedback_callback=self.undock_feedback_callback)
        self._send_goal_future.add_done_callback(self.undock_goal_response_callback)

    def undock_goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Undock goal rejected :(')
            return
        self.get_logger().info('Undock goal accepted :)')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.undock_result_callback)

    def undock_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Undock result: {result}')
        self.get_logger().info('Undocking complete! Starting navigation sequence...')
        self.send_next_goal()

    def undock_feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(f'Undock feedback: {feedback}')

    # ------------------------------------------------------------------
    # Main sequencer
    # ------------------------------------------------------------------

    def send_next_goal(self):

        # If we have visited all pickups, we are done
        if self.current_pickup_index >= len(PICKUP_LOCATIONS):
            self.celebrate()
            return

        if self.going_to_pickup:

            waypoint, final = PICKUP_LOCATIONS[self.current_pickup_index]
            x_final, y_final, oz_final, ow_final = final

            if self.nav_step == 'waypoint':
                # Step 1: Drive straight to the waypoint (no turn yet)
                self.get_logger().info(
                    f'Pickup {self.current_pickup_index + 1} — Step 1: waypoint at ({waypoint[0]}, {waypoint[1]})')
                self.nav_step = 'pre_position'
                self.send_goal(*waypoint)

            elif self.nav_step == 'pre_position':
                # Step 2: Navigate to PRE_POSITION_OFFSET short of the final y.
                # achieve_goal_heading=True so the robot SPINS HERE to the correct
                # final orientation — safely away from the piece.
                pre_y = y_final + PRE_POSITION_OFFSET  # back off along y
                self.get_logger().info(
                    f'Pickup {self.current_pickup_index + 1} — Step 2: pre-position at ({x_final}, {pre_y:.3f}), spinning to final heading')
                self.nav_step = 'final'
                self.send_goal(x_final, pre_y, oz_final, ow_final, achieve_heading=True)

            else:  # nav_step == 'final'
                # Step 3: Drive straight forward into the pickup — no reorientation.
                self.get_logger().info(
                    f'Pickup {self.current_pickup_index + 1} — Step 3: final straight approach ({FINAL_APPROACH_DIST} m)')
                self.nav_step = 'waypoint'  # reset for next pickup
                self.drive_to_pickup()

        else:  # going to dropoff

            if self.nav_step == 'waypoint':
                # Detect character and set dropoff coords
                character = self.detect_character()
                waypoint, final = DROPOFF_LOCATIONS[character]
                self.current_dropoff_coords = final
                self.get_logger().info(
                    f'Character detected: {character}. Dropoff waypoint at ({waypoint[0]}, {waypoint[1]})')
                self.nav_step = 'final'
                self.send_goal(*waypoint)

            else:  # nav_step == 'final'
                final = self.current_dropoff_coords
                self.get_logger().info(
                    f'Dropoff — final position at ({final[0]}, {final[1]})')
                self.nav_step = 'waypoint'  # reset for next cycle
                self.send_goal(*final)

    # ------------------------------------------------------------------
    # NavigateToPosition goal
    # ------------------------------------------------------------------

    def send_goal(self, x, y, oz, ow, achieve_heading=True):

        goal_msg = NavigateToPosition.Goal()
        goal_msg.goal_pose.header.frame_id = 'odom'
        goal_msg.goal_pose.pose.position.x = x
        goal_msg.goal_pose.pose.position.y = y
        goal_msg.goal_pose.pose.position.z = 0.0
        goal_msg.goal_pose.pose.orientation.x = 0.0
        goal_msg.goal_pose.pose.orientation.y = 0.0
        goal_msg.goal_pose.pose.orientation.z = oz
        goal_msg.goal_pose.pose.orientation.w = ow

        # When achieve_heading=False the robot stops at the position without
        # spinning — useful for waypoints where heading doesn't matter yet.
        goal_msg.achieve_goal_heading = achieve_heading

        self._action_client.wait_for_server()
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg, feedback_callback=self.feedback_callback)
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected :(')
            return
        self.get_logger().info('Goal accepted :)')
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        result = future.result().result
        self.get_logger().info(f'Navigation result: {result}')

        if self.going_to_pickup and self.nav_step == 'final':
            # About to do the straight drive — send_next_goal handles it
            self.send_next_goal()

        elif self.going_to_pickup and self.nav_step == 'waypoint':
            # Just finished the pre-position step; DriveDistance was already
            # called directly via drive_to_pickup(), so nothing to do here.
            pass

        elif not self.going_to_pickup and self.nav_step == 'waypoint':
            # Just arrived at the final dropoff position
            self.put_down()
            self.get_logger().info(
                f'Dropoff complete for pickup {self.current_pickup_index + 1}!')
            self.current_pickup_index += 1
            self.going_to_pickup = True
            self.send_next_goal()

        else:
            # Arrived at a waypoint — continue sequence
            self.send_next_goal()

    # ------------------------------------------------------------------
    # DriveDistance — straight final approach for pickups
    # ------------------------------------------------------------------

    def drive_to_pickup(self):
        goal_msg = DriveDistance.Goal()
        goal_msg.distance = FINAL_APPROACH_DIST
        goal_msg.max_translation_speed = FINAL_APPROACH_SPEED

        self._drive_client.wait_for_server()
        future = self._drive_client.send_goal_async(goal_msg)
        future.add_done_callback(self.drive_response_callback)

    def drive_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('DriveDistance goal rejected!')
            rclpy.shutdown()
            return
        self.get_logger().info('DriveDistance accepted — moving straight to piece.')
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.drive_result_callback)

    def drive_result_callback(self, future):
        self.get_logger().info('Reached pickup position — activating gripper.')
        self.pick_up()
        self.going_to_pickup = False
        self.send_next_goal()

    # ------------------------------------------------------------------
    # Feedback
    # ------------------------------------------------------------------

    def feedback_callback(self, feedback_msg):
        feedback = feedback_msg.feedback
        self.get_logger().info(f'Remaining distance: {feedback.remaining_travel_distance}')

    # ------------------------------------------------------------------
    # Character detection
    # ------------------------------------------------------------------

    def detect_character(self):

        self.get_logger().info('Detecting character...')

        scores = {}
        for name in self.class_names:
            scores[name.strip()[2:]] = 0.0

        num_frames = 10
        for i in range(num_frames):
            frame = self.picam2.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
            image = np.asarray(frame, dtype=np.float32).reshape(1, 224, 224, 3)
            image = (image / 127.5) - 1
            prediction = self.model.predict(image, verbose=0)
            for j, name in enumerate(self.class_names):
                class_name = name.strip()[2:]
                scores[class_name] += prediction[0][j]

        character = max(scores, key=scores.get)
        confidence = scores[character] / num_frames * 100
        self.get_logger().info(f'Detected: {character} ({confidence:.1f}% confidence)')
        return character

    # ------------------------------------------------------------------
    # Gripper actions
    # ------------------------------------------------------------------

    def pick_up(self):
        gripper_close()
        self.get_logger().info('Picking up character...')

    def put_down(self):
        gripper_open()
        self.get_logger().info('Putting down character...')

    def celebrate(self):
        self.get_logger().info('All characters delivered! Task complete!')
        gripper_cleanup()
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    robot = StrangerThingsRobot()
    rclpy.spin(robot)


if __name__ == '__main__':
    main()