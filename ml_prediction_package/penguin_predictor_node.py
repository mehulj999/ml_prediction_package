#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import joblib
import numpy as np
import os
import json
from ament_index_python.packages import get_package_share_directory

from std_msgs.msg import Float64MultiArray, String


class PenguinPredictorNode(Node):
    def __init__(self):
        super().__init__('penguin_predictor_node')

        # Get path to package models directory
        package_share_directory = get_package_share_directory('ml_prediction_package')
        model_path = os.path.join(package_share_directory, 'models', 'penguin_model.pkl')
        scaler_path = os.path.join(package_share_directory, 'models', 'penguin_scaler.pkl')

        # Load the trained model
        try:
            self.model = joblib.load(model_path)
            self.get_logger().info('Penguin model loaded successfully')
        except Exception as e:
            self.get_logger().error(f'Failed to load penguin model: {str(e)}')
            return

        # Load the scaler
        try:
            self.scaler = joblib.load(scaler_path)
            self.get_logger().info('Scaler loaded successfully')
        except Exception as e:
            self.get_logger().error(f'Failed to load scaler: {str(e)}')
            return

        # Species mapping
        self.species_map = {0: 'Adelie', 1: 'Chinstrap', 2: 'Gentoo'}

        # Create subscriber and publisher
        self.subscription = self.create_subscription(
            Float64MultiArray,
            'penguin_features',
            self.predict_callback,
            10
        )

        self.publisher = self.create_publisher(
            String,
            'penguin_prediction',
            10
        )

        self.get_logger().info('Penguin Predictor Node initialized')
        self.get_logger().info('Send penguin features to topic: penguin_features')
        self.get_logger().info('Format: [culmen_length_mm, culmen_depth_mm, flipper_length_mm, body_mass_g, island_encoded, sex_encoded]')

    def predict_callback(self, msg):
        try:
            expected_features = 6
            if len(msg.data) != expected_features:
                self.get_logger().error(f'Expected {expected_features} features, got {len(msg.data)}')
                return

            # Convert input to numpy array and reshape for a single sample
            features = np.array([msg.data])

            # Scale input features
            features_scaled = self.scaler.transform(features)

            # Log scaled features
            self.get_logger().info(f'Scaled features: {features_scaled}')

            # Predict using the trained model
            prediction = self.model.predict(features_scaled)[0]

            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(features_scaled)[0]
                confidence = float(np.max(probabilities))
            else:
                confidence = 1.0

            result = {
                'predicted_class': int(prediction),
                'species_name': self.species_map.get(prediction, f'Unknown_Class_{prediction}'),
                'confidence': confidence,
                'input_features': {
                    'culmen_length_mm': msg.data[0],
                    'culmen_depth_mm': msg.data[1],
                    'flipper_length_mm': msg.data[2],
                    'body_mass_g': msg.data[3],
                    'island_encoded': msg.data[4],
                    'sex_encoded': msg.data[5]
                }
            }

            # Publish the result
            response_msg = String()
            response_msg.data = json.dumps(result)
            self.publisher.publish(response_msg)

            self.get_logger().info(
                f'Penguin prediction: {result["species_name"]} '
                f'(class: {result["predicted_class"]}, '
                f'confidence: {result["confidence"]:.3f})'
            )

        except Exception as e:
            self.get_logger().error(f'Prediction error: {str(e)}')


def main(args=None):
    rclpy.init(args=args)
    node = PenguinPredictorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
