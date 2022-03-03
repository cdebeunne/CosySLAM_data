import rosbag
import rospy
import cv2
import sys
from cv_bridge import CvBridge

"""
A simple script to write images from a rosbag as png

arg 1 : rosbag path
"""

if __name__ == '__main__':

    rosbag_path = sys.argv[1]
    bridge = CvBridge()
    bag = rosbag.Bag(rosbag_path, "r")

    count = 0
    for _, msg, t in bag.read_messages(topics=['/camera/color/image_raw']):
        cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        im_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

        cv2.imwrite('%04i.png' % count, im_rgb)
        count += 1
