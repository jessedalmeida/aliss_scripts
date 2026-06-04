# Tracked Keypoints for Needle Tracking
geometry_msgs/Point32 needle_tip
geometry_msgs/Point32 needle_tail
geometry_msgs/Point32 left_arm_tip
geometry_msgs/Point32 right_arm_tip

# Per-keypoint visibility flag (parallel to the points above).
#   1 = visible / usable: the point coordinates are reliable.
#   0 = not visible: the keypoint is occluded or unlabeled and its
#       coordinates are NOT reliable (ignore / mask in any consumer).
uint8 needle_tip_visibility
uint8 needle_tail_visibility
uint8 left_arm_tip_visibility
uint8 right_arm_tip_visibility

# Bounding box for the needle in the image
# The 2D position (in pixels) and orientation of the bounding box center.
geometry_msgs/Pose2D bounding_box_center

# The size (in pixels) of the bounding box surrounding the object relative
#   to the pose of its center.
float64 bounding_box_size_x
float64 bounding_box_size_y
