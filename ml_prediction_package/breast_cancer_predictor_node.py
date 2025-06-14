#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import joblib
import numpy as np
import os
import json
from ament_index_python.packages import get_package_share_directory

from std_msgs.msg import Float64MultiArray, String


class BreastCancerPredictorNode(Node):
    def __init__(self):
        super().__init__('breast_cancer_predictor_node')

        # Load the trained model
        package_share_directory = get_package_share_directory('ml_prediction_package')
        model_path = os.path.join(package_share_directory, 'models', 'breast_cancer_model.pkl')

        try:
            self.model = joblib.load(model_path)
            self.get_logger().info('Breast cancer model loaded successfully')
        except Exception as e:
            self.get_logger().error(f'Failed to load breast cancer model: {str(e)}')
            return

        # Diagnosis mapping
        self.diagnosis_map = {0: 'Benign', 1: 'Malignant'}

        # Create subscriber and publisher
        self.subscription = self.create_subscription(
            Float64MultiArray,
            'breast_cancer_features',
            self.predict_callback,
            10
        )

        self.publisher = self.create_publisher(
            String,
            'breast_cancer_prediction',
            10
        )

        self.get_logger().info('Breast Cancer Predictor Node initialized')
        self.get_logger().info('Send breast cancer features to topic: breast_cancer_features')
        self.get_logger().info('Format: [radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean, radius_se, texture_se, perimeter_se, area_se, smoothness_se, compactness_se, concavity_se, concave_points_se, symmetry_se, fractal_dimension_se, radius_worst, texture_worst, perimeter_worst, area_worst, smoothness_worst, compactness_worst, concavity_worst, concave_points_worst, symmetry_worst, fractal_dimension_worst]')

    def predict_callback(self, msg):
        try:
            # Check if we have exactly 30 features (mean, se, and worst for each measurement)
            if len(msg.data) != 30:
                self.get_logger().error('Expected 30 features, got {}'.format(len(msg.data)))
                return

            # Prepare input data
            features = np.array([msg.data])

            # Log the input features for debugging
            self.get_logger().info(f'Input features shape: {features.shape}')
            self.get_logger().info(f'Feature ranges - Min: {np.min(features):.4f}, Max: {np.max(features):.4f}')

            # Make prediction
            prediction = self.model.predict(features)[0]

            # Log prediction details
            self.get_logger().info(f'Raw prediction: {prediction}')

            # Get prediction probabilities for confidence
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(features)[0]
                confidence = float(np.max(probabilities))
            else:
                confidence = 1.0  # Default confidence if probabilities not available

            # Create response as JSON string
            result = {
                'predicted_class': int(prediction),
                'diagnosis': self.diagnosis_map[prediction],
                'confidence': confidence,
                'input_features': {
                    'radius_mean': msg.data[0],
                    'texture_mean': msg.data[1],
                    'perimeter_mean': msg.data[2],
                    'area_mean': msg.data[3],
                    'smoothness_mean': msg.data[4],
                    'compactness_mean': msg.data[5],
                    'concavity_mean': msg.data[6],
                    'concave_points_mean': msg.data[7],
                    'symmetry_mean': msg.data[8],
                    'fractal_dimension_mean': msg.data[9],
                    'radius_se': msg.data[10],
                    'texture_se': msg.data[11],
                    'perimeter_se': msg.data[12],
                    'area_se': msg.data[13],
                    'smoothness_se': msg.data[14],
                    'compactness_se': msg.data[15],
                    'concavity_se': msg.data[16],
                    'concave_points_se': msg.data[17],
                    'symmetry_se': msg.data[18],
                    'fractal_dimension_se': msg.data[19],
                    'radius_worst': msg.data[20],
                    'texture_worst': msg.data[21],
                    'perimeter_worst': msg.data[22],
                    'area_worst': msg.data[23],
                    'smoothness_worst': msg.data[24],
                    'compactness_worst': msg.data[25],
                    'concavity_worst': msg.data[26],
                    'concave_points_worst': msg.data[27],
                    'symmetry_worst': msg.data[28],
                    'fractal_dimension_worst': msg.data[29]
                }
            }

            # Publish the response
            response_msg = String()
            response_msg.data = json.dumps(result)
            self.publisher.publish(response_msg)

            self.get_logger().info(
                f'Breast cancer prediction: {result["diagnosis"]} '
                f'(class: {result["predicted_class"]}, '
                f'confidence: {result["confidence"]:.3f})'
            )

        except Exception as e:
            self.get_logger().error(f'Prediction error: {str(e)}')


def main(args=None):
    rclpy.init(args=args)
    node = BreastCancerPredictorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()