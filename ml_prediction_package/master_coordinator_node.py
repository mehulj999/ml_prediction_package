
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32
import json
import time

class TrainingCoordinatorNode(Node):
    def __init__(self):
        super().__init__('training_coordinator_node')

        # Publishers to start training
        self.iris_start_pub = self.create_publisher(String, '/training/iris/start', 10)
        self.breast_cancer_start_pub = self.create_publisher(String, '/training/breast_cancer/start', 10)
        self.penguin_start_pub = self.create_publisher(String, '/training/penguin/start', 10)
        self.fruit_start_pub = self.create_publisher(String, '/training/fruit/start', 10)

        # Subscribers for results
        self.iris_status_sub = self.create_subscription(String, '/training/iris/status', self.iris_status_callback, 10)
        self.breast_cancer_status_sub = self.create_subscription(String, '/training/breast_cancer/status', self.breast_cancer_status_callback, 10)
        self.penguin_status_sub = self.create_subscription(String, '/training/penguin/status', self.penguin_status_callback, 10)
        self.fruit_status_sub = self.create_subscription(String, '/training/fruit/status', self.fruit_status_callback, 10)

        # Results subscribers
        self.iris_results_sub = self.create_subscription(String, '/training/iris/results', self.iris_results_callback, 10)
        self.breast_cancer_results_sub = self.create_subscription(String, '/training/breast_cancer/results', self.breast_cancer_results_callback, 10)
        self.penguin_results_sub = self.create_subscription(String, '/training/penguin/results', self.penguin_results_callback, 10)
        self.fruit_results_sub = self.create_subscription(String, '/training/fruit/results', self.fruit_results_callback, 10)

        # Status tracking
        self.training_status = {
            'iris': 'IDLE',
            'breast_cancer': 'IDLE',
            'penguin': 'IDLE',
            'fruit': 'IDLE'
        }

        # Timer to start all training
        self.timer = self.create_timer(5.0, self.start_all_training)
        self.training_started = False

        self.get_logger().info('Training Coordinator Node initialized')

    def start_all_training(self):
        if not self.training_started:
            self.get_logger().info('Starting all training processes...')

            # Start all training nodes
            start_msg = String()
            start_msg.data = "START"

            self.iris_start_pub.publish(start_msg)
            self.breast_cancer_start_pub.publish(start_msg)
            self.penguin_start_pub.publish(start_msg)
            self.fruit_start_pub.publish(start_msg)

            self.training_started = True
            self.timer.cancel()  # Stop the timer

    def iris_status_callback(self, msg):
        self.training_status['iris'] = msg.data
        self.get_logger().info(f'Iris status: {msg.data}')

    def breast_cancer_status_callback(self, msg):
        self.training_status['breast_cancer'] = msg.data
        self.get_logger().info(f'Breast Cancer status: {msg.data}')

    def penguin_status_callback(self, msg):
        self.training_status['penguin'] = msg.data
        self.get_logger().info(f'Penguin status: {msg.data}')

    def fruit_status_callback(self, msg):
        self.training_status['fruit'] = msg.data
        self.get_logger().info(f'Fruit status: {msg.data}')

    def iris_results_callback(self, msg):
        results = json.loads(msg.data)
        self.get_logger().info(f'Iris Results - Accuracy: {results["accuracy"]:.4f}')

    def breast_cancer_results_callback(self, msg):
        results = json.loads(msg.data)
        self.get_logger().info(f'Breast Cancer Results - Accuracy: {results["accuracy"]:.4f}')

    def penguin_results_callback(self, msg):
        results = json.loads(msg.data)
        self.get_logger().info(f'Penguin Results - Accuracy: {results["accuracy"]:.4f}')

    def fruit_results_callback(self, msg):
        results = json.loads(msg.data)
        self.get_logger().info(f'Fruit Results - Accuracy: {results["accuracy"]:.4f}')

def main(args=None):
    rclpy.init(args=args)
    node = TrainingCoordinatorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()