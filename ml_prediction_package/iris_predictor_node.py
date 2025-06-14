#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import joblib
import numpy as np
import os
import json
from ament_index_python.packages import get_package_share_directory

from std_msgs.msg import Float64MultiArray, String


class IrisPredictorNode(Node):
    def __init__(self):
        super().__init__('iris_predictor_node')

        # Load the trained model
        package_share_directory = get_package_share_directory('ml_prediction_package')
        model_path = os.path.join(package_share_directory, 'models', 'iris_voting_classifier.pkl')

        try:
            self.model = joblib.load(model_path)
            self.get_logger().info('Iris model loaded successfully')
        except Exception as e:
            self.get_logger().error(f'Failed to load iris model: {str(e)}')
            return

        # Species mapping
        self.species_map = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}

        # Create subscriber and publisher
        self.subscription = self.create_subscription(
            Float64MultiArray,
            'iris_features',
            self.predict_callback,
            10
        )

        self.publisher = self.create_publisher(
            String,
            'iris_prediction',
            10
        )

        self.get_logger().info('Iris Predictor Node initialized')
        self.get_logger().info('Send iris features to topic: iris_features')
        self.get_logger().info('Format: [sepal_length, sepal_width, petal_length, petal_width]')

    def predict_callback(self, msg):
        try:
            # Check if we have exactly 4 features
            if len(msg.data) != 4:
                self.get_logger().error('Expected 4 features, got {}'.format(len(msg.data)))
                return

            # Prepare input data
            features = np.array([msg.data])

            # Make prediction
            prediction = self.model.predict(features)[0]

            # Get prediction probabilities for confidence
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(features)[0]
                confidence = float(np.max(probabilities))
            else:
                confidence = 1.0  # Default confidence if probabilities not available

            # Create response as JSON string
            result = {
                'predicted_class': int(prediction),
                'species_name': self.species_map[prediction],
                'confidence': confidence,
                'input_features': {
                    'sepal_length': msg.data[0],
                    'sepal_width': msg.data[1],
                    'petal_length': msg.data[2],
                    'petal_width': msg.data[3]
                }
            }

            # Publish the response
            response_msg = String()
            response_msg.data = json.dumps(result)
            self.publisher.publish(response_msg)

            self.get_logger().info(
                f'Iris prediction: {result["species_name"]} '
                f'(class: {result["predicted_class"]}, '
                f'confidence: {result["confidence"]:.3f})'
            )

        except Exception as e:
            self.get_logger().error(f'Prediction error: {str(e)}')


def main(args=None):
    rclpy.init(args=args)
    node = IrisPredictorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()