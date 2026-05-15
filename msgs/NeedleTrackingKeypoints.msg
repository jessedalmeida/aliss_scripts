# Tracked Keypoints for Needle Tracking
geometry_msgs/Point32 needle_tip
geometry_msgs/Point32 needle_tail
geometry_msgs/Point32 left_arm_tip
geometry_msgs/Point32 right_arm_tip

# Bounding box for the needle in the image
# The 2D position (in pixels) and orientation of the bounding box center.
geometry_msgs/Pose2D bounding_box_center

# The size (in pixels) of the bounding box surrounding the object relative
#   to the pose of its center.
float64 bounding_box_size_x
float64 bounding_box_size_y