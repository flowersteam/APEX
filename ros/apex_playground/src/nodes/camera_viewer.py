#!/usr/bin/python

import rospy
import matplotlib.pyplot as plt

def show_image():
    rospy.wait_for_service("camera")
    read = rospy.ServiceProxy("camera")
    image = read()
    plt.imshow(image)
    plt.show()

if __name__ == "__main__":
    show_image()