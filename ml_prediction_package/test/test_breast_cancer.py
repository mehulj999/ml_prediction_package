#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import time

def send_test_cases():
    rclpy.init()
    node = rclpy.create_node('breast_cancer_test_publisher')
    publisher = node.create_publisher(Float64MultiArray, 'breast_cancer_features', 10)

    # Wait for connection
    time.sleep(1)

    # Test case 1: Benign
    msg1 = Float64MultiArray()
    msg1.data = [13.54, 14.36, 87.46, 566.3, 0.09779, 0.08129, 0.06664, 0.04781, 0.1885, 0.05766,
                 0.2699, 0.7886, 2.058, 23.56, 0.008462, 0.0146, 0.02387, 0.01315, 0.0198, 0.0023,
                 15.11, 19.26, 99.7, 711.2, 0.144, 0.1773, 0.239, 0.1288, 0.2977, 0.07259]

    publisher.publish(msg1)
    node.get_logger().info('Benign test case sent!')

    time.sleep(2)

    # Test case 2: Malignant
    msg2 = Float64MultiArray()
    msg2.data = [20.57, 17.77, 132.90, 1326.0, 0.08474, 0.07864, 0.0869, 0.07017, 0.1812, 0.05667,
                 0.5435, 0.7339, 3.398, 74.08, 0.005225, 0.01308, 0.01860, 0.01340, 0.01389, 0.003532,
                 24.99, 23.41, 158.80, 1956.0, 0.1238, 0.1866, 0.2416, 0.1860, 0.2750, 0.08902]

    publisher.publish(msg2)
    node.get_logger().info('Malignant test case sent!')

    rclpy.shutdown()

if __name__ == '__main__':
    send_test_cases()