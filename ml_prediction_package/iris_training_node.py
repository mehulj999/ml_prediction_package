#!/usr/bin/env python3
"""
Iris Training Node for ROS2 - Fixed Path Resolution
Part of ml_prediction_package
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import json
import os
from ament_index_python.packages import get_package_share_directory

class IrisTrainingNode(Node):
    def __init__(self):
        super().__init__('iris_training_node')

        # Publishers
        self.status_pub = self.create_publisher(String, '/training/iris/status', 10)
        self.accuracy_pub = self.create_publisher(Float32, '/training/iris/accuracy', 10)
        self.results_pub = self.create_publisher(String, '/training/iris/results', 10)

        # Subscribers
        self.train_sub = self.create_subscription(
            String, '/training/iris/start', self.train_callback, 10
        )

        # Parameters with default paths relative to package
        self.declare_parameter('data_path', 'data/Iris.csv')
        self.declare_parameter('model_output_path', 'models/iris_voting_classifier.pkl')

        self.get_logger().info('Iris Training Node initialized')

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
        self.get_logger().info('Starting Iris model training...')

        # Publish status
        status_msg = String()
        status_msg.data = "TRAINING_STARTED"
        self.status_pub.publish(status_msg)

        try:
            # Get parameters and resolve paths
            data_path_param = self.get_parameter('data_path').get_parameter_value().string_value
            model_path_param = self.get_parameter('model_output_path').get_parameter_value().string_value

            # Resolve paths relative to package
            data_path = self.resolve_path(data_path_param)
            model_path = self.resolve_path(model_path_param)

            self.get_logger().info(f'Using data path: {data_path}')
            self.get_logger().info(f'Using model path: {model_path}')

            # Check if data file exists
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Data file not found: {data_path}")

            # Create models directory if it doesn't exist
            model_dir = os.path.dirname(model_path)
            if model_dir and not os.path.exists(model_dir):
                os.makedirs(model_dir, exist_ok=True)
                self.get_logger().info(f'Created model directory: {model_dir}')

            # Load and preprocess data
            self.get_logger().info(f'Loading data from: {data_path}')
            df = pd.read_csv(data_path)
            self.get_logger().info(f'Data loaded successfully. Shape: {df.shape}')

            if 'Id' in df.columns:
                df.drop('Id', axis=1, inplace=True)

            le = LabelEncoder()
            df['Species'] = le.fit_transform(df['Species'])

            X = df.drop('Species', axis=1)
            y = df['Species']

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            self.get_logger().info(f'Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}')

            # Create individual models
            knn = KNeighborsClassifier(n_neighbors=3)
            dt = DecisionTreeClassifier(random_state=42)
            rf = RandomForestClassifier(n_estimators=100, random_state=42)

            # Ensemble Voting Classifier
            voting_clf = VotingClassifier(estimators=[
                ('knn', knn),
                ('dt', dt),
                ('rf', rf)
            ], voting='hard')

            self.get_logger().info('Training ensemble model...')

            # Train model
            voting_clf.fit(X_train, y_train)

            # Evaluate
            y_pred = voting_clf.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(
                y_test, y_pred, target_names=le.classes_, output_dict=True
            )

            # Save model
            joblib.dump(voting_clf, model_path)
            self.get_logger().info(f'Model saved to: {model_path}')

            # Publish results
            accuracy_msg = Float32()
            accuracy_msg.data = float(accuracy)
            self.accuracy_pub.publish(accuracy_msg)

            results_msg = String()
            results_msg.data = json.dumps({
                'accuracy': accuracy,
                'classification_report': report,
                'model_path': model_path,
                'dataset_shape': df.shape,
                'features': X.columns.tolist(),
                'classes': le.classes_.tolist()
            })
            self.results_pub.publish(results_msg)

            status_msg.data = "TRAINING_COMPLETED"
            self.status_pub.publish(status_msg)

            self.get_logger().info(f'Iris training completed with accuracy: {accuracy:.4f}')

        except FileNotFoundError as e:
            error_msg = f"Data file not found: {str(e)}"
            self.get_logger().error(error_msg)
            status_msg.data = f"TRAINING_FAILED: {error_msg}"
            self.status_pub.publish(status_msg)

        except Exception as e:
            error_msg = f"Training failed: {str(e)}"
            self.get_logger().error(error_msg)
            status_msg.data = f"TRAINING_FAILED: {error_msg}"
            self.status_pub.publish(status_msg)

def main(args=None):
    rclpy.init(args=args)
    node = IrisTrainingNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()