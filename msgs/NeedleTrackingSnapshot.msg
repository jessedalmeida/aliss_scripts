# Rectified image from the camera
sensor_msgs/Image img

# Segmentation mask of the needle
sensor_msgs/Image mask

# Arm state (e.g., joint positions, velocities, etc.)
# joint value of arm
sensor_msgs/JointState right_arm_jp
sensor_msgs/JointState left_arm_jp

# original arm control pose
geometry_msgs/PoseStamped right_arm_cp
geometry_msgs/PoseStamped left_arm_cp

# tip pose from ves_smoother
geometry_msgs/PoseWithCovarianceStamped right_tip_pose
geometry_msgs/PoseWithCovarianceStamped left_tip_pose


# Tracked keypoints (e.g., needle tip, needle tail, arm tips)
NeedleTrackingKeypoints keypoints

# Timestamp for synchronization
builtin_interfaces/Time stamp