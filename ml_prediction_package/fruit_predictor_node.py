#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import joblib
import os
import json
import numpy as np
import pandas as pd
from ament_index_python.packages import get_package_share_directory

from std_msgs.msg import Float64MultiArray, String


class FruitPredictorNode(Node):
    def __init__(self):
        super().__init__('fruit_predictor_node')

        # Load trained model pipeline and label encoder
        package_dir = get_package_share_directory('ml_prediction_package')
        model_path = os.path.join(package_dir, 'models', 'fruit_model_with_color.pkl')
        encoder_path = os.path.join(package_dir, 'models', 'fruit_label_encoder.pkl')

        try:
            self.model = joblib.load(model_path)
            self.label_encoder = joblib.load(encoder_path)
            self.get_logger().info('Model and label encoder loaded successfully')
        except Exception as e:
            self.get_logger().error(f'Failed to load model or encoder: {str(e)}')
            return

        # Create subscriber and publisher
        self.subscription = self.create_subscription(
            Float64MultiArray,
            'fruit_features',
            self.predict_callback,
            10
        )


        self.publisher = self.create_publisher(
            String,
            'fruit_prediction',
            10
        )

        self.get_logger().info('Fruit Predictor Node initialized')
        self.get_logger().info('Publish JSON to topic: fruit_features')
        self.get_logger().info('Example format: {"Color": "Red", "Weight": 240, "Sphericity": 0.45}')
    def predict_callback(self, msg):
        try:
            # Expecting: [color_index, weight, sphericity]
            if len(msg.data) != 3:
                self.get_logger().error(f'Expected 3 values: color_index, weight, sphericity — got {len(msg.data)}')
                return

            color_index = int(msg.data[0])
            weight = msg.data[1]
            sphericity = msg.data[2]

            # Map index back to color name (must match training data)
            color_list = self.model.named_steps['preprocessing'].transformers_[1][1].categories_[0]
            if color_index < 0 or color_index >= len(color_list):
                self.get_logger().error(f'Invalid color index: {color_index}. Valid indices: 0 to {len(color_list) - 1}')
                return

            color_str = color_list[color_index]

            input_df = pd.DataFrame([{
                'Color': color_str,
                'Weight': weight,
                'Sphericity': sphericity
            }])

            prediction = self.model.predict(input_df)[0]
            confidence = float(np.max(self.model.predict_proba(input_df)[0]))

            class_name = self.label_encoder.inverse_transform([prediction])[0]

            result = {
                'predicted_class': int(prediction),
                'fruit_name': class_name,
                'confidence': confidence
            }

            msg_out = String()
            msg_out.data = json.dumps(result)
            self.publisher.publish(msg_out)

            self.get_logger().info(f'Predicted: {class_name} (class {prediction}, confidence {confidence:.3f})')

        except Exception as e:
            self.get_logger().error(f'Prediction failed: {str(e)}')


def main(args=None):
    rclpy.init(args=args)
    node = FruitPredictorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
