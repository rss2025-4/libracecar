import numpy as np
import tf_transformations
from geometry_msgs.msg import Pose, Transform, TransformStamped
from rclpy.time import Time


def pose_to_transform(p: Pose) -> Transform:
    ans = Transform()
    ans.translation.x = p.position.x
    ans.translation.y = p.position.y
    ans.translation.z = p.position.z
    ans.rotation = p.orientation
    return ans


def transform_to_pose(t: Transform) -> Pose:
    ans = Pose()
    ans.position.x = t.translation.x
    ans.position.y = t.translation.y
    ans.position.z = t.translation.z
    ans.orientation = t.rotation
    return ans


# tf_to_se3 and se3_to_tf are copied and modified from tf_lecture_example


def tf_to_se3(transform: Transform) -> np.ndarray:
    """
    Convert a TransformStamped message to a 4x4 SE3 matrix
    """
    q = transform.rotation
    q = [q.x, q.y, q.z, q.w]
    t = transform.translation
    mat = tf_transformations.quaternion_matrix(q)
    mat[0, 3] = t.x
    mat[1, 3] = t.y
    mat[2, 3] = t.z
    return mat


def se3_to_tf(mat: np.ndarray, time: Time, parent: str, child: str) -> TransformStamped:
    """
    Convert a 4x4 SE3 matrix to a TransformStamped message
    """
    obj = TransformStamped()

    # current time
    obj.header.stamp = time.to_msg()

    # frame names
    obj.header.frame_id = parent
    obj.child_frame_id = child

    # translation component
    obj.transform.translation.x = mat[0, 3]
    obj.transform.translation.y = mat[1, 3]
    obj.transform.translation.z = mat[2, 3]

    # rotation (quaternion)
    q = tf_transformations.quaternion_from_matrix(mat)
    obj.transform.rotation.x = q[0]
    obj.transform.rotation.y = q[1]
    obj.transform.rotation.z = q[2]
    obj.transform.rotation.w = q[3]

    return obj


def pose_2d(x: float, y: float, orientation: float) -> Pose:
    # from test_wall_follower.py
    p = Pose()

    p.position.x = x
    p.position.y = y

    # Convert theta to a quaternion
    quaternion = tf_transformations.quaternion_from_euler(0, 0, orientation)
    p.orientation.y = quaternion[1]
    p.orientation.z = quaternion[2]
    p.orientation.w = quaternion[3]

    return p
