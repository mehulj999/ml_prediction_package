#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import json
import os
from ament_index_python.packages import get_package_share_directory

class FruitTrainingNode(Node):
    def __init__(self):
        super().__init__('fruit_training_node')

        # Publishers
        self.status_pub = self.create_publisher(String, '/training/fruit/status', 10)
        self.accuracy_pub = self.create_publisher(Float32, '/training/fruit/accuracy', 10)
        self.results_pub = self.create_publisher(String, '/training/fruit/results', 10)

        # Subscribers
        self.train_sub = self.create_subscription(String, '/training/fruit/start', self.train_callback, 10)

        # Parameters
        self.declare_parameter('data_path', 'data/fruits_weight_sphercity.csv')
        self.declare_parameter('model_output_path', 'models/fruit_model_with_color.pkl')
        self.declare_parameter('encoder_output_path', 'models/fruit_label_encoder.pkl')

        self.get_logger().info('Fruit Training Node initialized')

    def resolve_path(self, path):
      """Resolve path relative to package if it's not absolute"""
      if os.path.isabs(path):
          return path
      else:
          # Get package share directory and join with relative path
          package_share_dir = get_package_share_directory('ml_prediction_package')
          resolved_path = os.path.join(package_share_dir, path)
          self.get_logger().info(f'Resolved path: {path} -> {resolved_path}')
          return resolved_path

    def train_callback(self, msg):
        self.get_logger().info('Starting Fruit model training...')

        status_msg = String()
        status_msg.data = "TRAINING_STARTED"
        self.status_pub.publish(status_msg)

        try:
            # Get parameters
            data_path = self.resolve_path(self.get_parameter('data_path').get_parameter_value().string_value)
            model_path = self.resolve_path(self.get_parameter('model_output_path').get_parameter_value().string_value)
            encoder_path = self.resolve_path(self.get_parameter('encoder_output_path').get_parameter_value().string_value)
            self.get_logger().info(f'Using data path: {data_path}')
            self.get_logger().info(f'Using model path: {model_path}')

            # Check if data file exists
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Data file not found: {data_path}")

            # Load and preprocess data
            df = pd.read_csv(data_path)

            # Encode target labels
            le = LabelEncoder()
            df['label_encoded'] = le.fit_transform(df['labels'])

            # Define features and label
            X = df[['Color', 'Weight', 'Sphericity']]
            y = df['label_encoded']

            # Define preprocessing for column types
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', StandardScaler(), ['Weight', 'Sphericity']),
                    ('cat', OneHotEncoder(), ['Color'])
                ]
            )

            # Pipeline: preprocessing + classifier
            model_pipeline = Pipeline([
                ('preprocessing', preprocessor),
                ('classifier', LogisticRegression(class_weight='balanced'))
            ])

            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            # Train model
            model_pipeline.fit(X_train, y_train)

            # Evaluate
            y_pred = model_pipeline.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True)

            # Save model and label encoder
            joblib.dump(model_pipeline, model_path)
            joblib.dump(le, encoder_path)

            # Publish results
            accuracy_msg = Float32()
            accuracy_msg.data = float(accuracy)
            self.accuracy_pub.publish(accuracy_msg)

            results_msg = String()
            results_msg.data = json.dumps({
                'accuracy': accuracy,
                'classification_report': report,
                'model_path': model_path,
                'encoder_path': encoder_path
            })
            self.results_pub.publish(results_msg)

            status_msg.data = "TRAINING_COMPLETED"
            self.status_pub.publish(status_msg)

            self.get_logger().info(f'Fruit training completed with accuracy: {accuracy:.4f}')

        except Exception as e:
            self.get_logger().error(f'Training failed: {str(e)}')
            status_msg.data = f"TRAINING_FAILED: {str(e)}"
            self.status_pub.publish(status_msg)

def main(args=None):
    rclpy.init(args=args)
    node = FruitTrainingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()