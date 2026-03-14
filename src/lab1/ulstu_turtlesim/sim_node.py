import math
import time
import traceback

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from turtlesim.msg import Pose
from turtlesim.srv import SetPen, TeleportAbsolute


class SimNode(Node):
    def __init__(self):
        try:
            super().__init__('sim_node')

            self._pose = None
            self._tick = 0
            self._trajectory_completed = False

            self._trajectory_start = (1.0, 1.0)
            self._trajectory_end = (10.0, 7.5)
            self._trajectory_curve_factor = 18.0
            self._trajectory_samples = 240
            self._trajectory_reach_distance = 0.12
            self._trajectory_finish_distance = 0.03
            self._trajectory_linear_speed = 1.6
            self._trajectory_min_linear_speed = 0.08
            self._trajectory_angular_gain = 6.0
            self._trajectory_max_angular_speed = 4.0
            self._service_wait_timeout = 10.0

            self._trajectory_points = self._build_logarithmic_path(
                self._trajectory_start,
                self._trajectory_end,
            )
            self._trajectory_index = 1

            self._twist_publisher = self.create_publisher(Twist, '/turtle1/cmd_vel', 10)
            self.create_subscription(Pose, '/turtle1/pose', self._pose_callback, 10)
            self._set_pen_client = self.create_client(SetPen, '/turtle1/set_pen')
            self._teleport_client = self.create_client(
                TeleportAbsolute,
                '/turtle1/teleport_absolute',
            )

            if self._prepare_trajectory_start():
                self.get_logger().info(
                    'The turtle was moved to the lower-left corner '
                    'and is ready to start the logarithmic trajectory.'
                )
            else:
                self.get_logger().warning(
                    'Failed to prepare the start point. '
                    'The turtle will stay still until the services become available.'
                )
                self._trajectory_completed = True

            self.create_timer(0.05, self._node_callback)
        except Exception as err:
            self.get_logger().error(
                ''.join(traceback.TracebackException.from_exception(err).format())
            )

    def _pose_callback(self, msg):
        self._pose = msg

    def _node_callback(self):
        try:
            move_cmd = Twist()

            if not self._trajectory_completed and self._pose is not None:
                move_cmd = self._compute_trajectory_command()

            self._twist_publisher.publish(move_cmd)
            self._tick += 1
        except Exception as err:
            self.get_logger().error(
                ''.join(traceback.TracebackException.from_exception(err).format())
            )

    def _prepare_trajectory_start(self):
        start_heading = self._heading_to_point(
            self._trajectory_points[0],
            self._trajectory_points[1],
        )

        if not self._set_pen_enabled(False):
            return False

        if not self._teleport_to_pose(
            self._trajectory_start[0],
            self._trajectory_start[1],
            start_heading,
        ):
            self._set_pen_enabled(True)
            return False

        if not self._set_pen_enabled(True):
            return False

        self._set_local_pose(
            self._trajectory_start[0],
            self._trajectory_start[1],
            start_heading,
        )
        return True

    def _compute_trajectory_command(self):
        move_cmd = Twist()

        self._advance_trajectory_index()
        finish_distance = self._distance_to_point(self._trajectory_points[-1])

        if (
            self._trajectory_index == len(self._trajectory_points) - 1
            and finish_distance <= self._trajectory_finish_distance
        ):
            self._trajectory_completed = True
            self.get_logger().info('The logarithmic trajectory is complete.')
            return move_cmd

        target_point = self._trajectory_points[self._trajectory_index]
        target_distance = self._distance_to_point(target_point)
        desired_heading = self._heading_to_point((self._pose.x, self._pose.y), target_point)
        heading_error = self._normalize_angle(desired_heading - self._pose.theta)

        angular_speed = max(
            -self._trajectory_max_angular_speed,
            min(
                self._trajectory_max_angular_speed,
                self._trajectory_angular_gain * heading_error,
            ),
        )

        if abs(heading_error) > 1.2:
            linear_speed = 0.0
        else:
            linear_speed = min(
                self._trajectory_linear_speed,
                0.3 + target_distance * 2.0,
                0.12 + finish_distance * 2.5,
            )
            linear_speed *= max(0.2, math.cos(heading_error))
            linear_speed = max(self._trajectory_min_linear_speed, linear_speed)

        move_cmd.linear.x = linear_speed
        move_cmd.angular.z = angular_speed
        return move_cmd

    def _advance_trajectory_index(self):
        while self._trajectory_index < len(self._trajectory_points) - 1:
            target_point = self._trajectory_points[self._trajectory_index]
            if self._distance_to_point(target_point) >= self._trajectory_reach_distance:
                break
            self._trajectory_index += 1

    def _build_logarithmic_path(self, start_point, end_point):
        start_x, start_y = start_point
        end_x, end_y = end_point

        if end_x <= start_x or end_y <= start_y:
            raise ValueError('The end point must be above and to the right of the start point.')

        path = []
        x_span = end_x - start_x
        y_span = end_y - start_y
        denominator = math.log1p(self._trajectory_curve_factor)

        for sample_index in range(self._trajectory_samples + 1):
            ratio = sample_index / self._trajectory_samples
            x_coord = start_x + x_span * ratio
            y_coord = start_y + y_span * math.log1p(
                self._trajectory_curve_factor * ratio
            ) / denominator
            path.append((x_coord, y_coord))

        return path

    def _set_pen_enabled(self, enabled):
        if not self._wait_for_service(self._set_pen_client, '/turtle1/set_pen'):
            return False

        request = SetPen.Request()
        request.r = 255
        request.g = 255
        request.b = 255
        request.width = 3
        request.off = 0 if enabled else 1

        future = self._set_pen_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        return future.result() is not None

    def _teleport_to_pose(self, x_coord, y_coord, theta):
        if not self._wait_for_service(
            self._teleport_client,
            '/turtle1/teleport_absolute',
        ):
            return False

        request = TeleportAbsolute.Request()
        request.x = x_coord
        request.y = y_coord
        request.theta = theta

        future = self._teleport_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        return future.result() is not None

    def _set_local_pose(self, x_coord, y_coord, theta):
        pose = Pose()
        pose.x = x_coord
        pose.y = y_coord
        pose.theta = theta
        pose.linear_velocity = 0.0
        pose.angular_velocity = 0.0
        self._pose = pose

    def _distance_to_point(self, point):
        return math.dist((self._pose.x, self._pose.y), point)

    def _wait_for_service(self, client, service_name):
        deadline = time.monotonic() + self._service_wait_timeout

        while rclpy.ok() and not client.wait_for_service(timeout_sec=0.5):
            if time.monotonic() >= deadline:
                self.get_logger().warning(f'{service_name} is unavailable')
                return False

        return True

    @staticmethod
    def _heading_to_point(start_point, end_point):
        return math.atan2(end_point[1] - start_point[1], end_point[0] - start_point[0])

    @staticmethod
    def _normalize_angle(angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle


def main(args=None):
    try:
        rclpy.init(args=args)
        sim_node = SimNode()
        rclpy.spin(sim_node)
        sim_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    except KeyboardInterrupt:
        if rclpy.ok():
            rclpy.shutdown()
    except Exception as err:
        print(''.join(traceback.TracebackException.from_exception(err).format()))
