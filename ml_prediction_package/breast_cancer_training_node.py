#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import json
import os
from ament_index_python.packages import get_package_share_directory

class BreastCancerTrainingNode(Node):
    def __init__(self):
        super().__init__('breast_cancer_training_node')

        # Publishers
        self.status_pub = self.create_publisher(String, '/training/breast_cancer/status', 10)
        self.accuracy_pub = self.create_publisher(Float32, '/training/breast_cancer/accuracy', 10)
        self.results_pub = self.create_publisher(String, '/training/breast_cancer/results', 10)

        # Subscribers
        self.train_sub = self.create_subscription(String, '/training/breast_cancer/start', self.train_callback, 10)

        # Parameters
        self.declare_parameter('data_path', 'data/Breast_Cancer.csv')
        self.declare_parameter('model_output_path', 'models/breast_cancer_model.pkl')

        self.get_logger().info('Breast Cancer Training Node initialized')


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
        self.get_logger().info('Starting Breast Cancer model training...')

        status_msg = String()
        status_msg.data = "TRAINING_STARTED"
        self.status_pub.publish(status_msg)

        try:
            # Get parameters
            data_path = self.resolve_path(self.get_parameter('data_path').get_parameter_value().string_value)
            model_path = self.resolve_path(self.get_parameter('model_output_path').get_parameter_value().string_value)

            # Load and preprocess data
            df = pd.read_csv(data_path)
            if 'id' in df.columns:
                df.drop('id', axis=1, inplace=True)

            le = LabelEncoder()
            df['diagnosis'] = le.fit_transform(df['diagnosis'])

            X = df.drop('diagnosis', axis=1)
            y = df['diagnosis']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Create pipeline with StandardScaler and SVC
            pipeline = make_pipeline(StandardScaler(), SVC(kernel='linear', probability=True, random_state=42))

            # Train model
            pipeline.fit(X_train, y_train)

            # Evaluate
            y_pred = pipeline.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            conf_matrix = confusion_matrix(y_test, y_pred)

            # Save model
            joblib.dump(pipeline, model_path)

            # Publish results
            accuracy_msg = Float32()
            accuracy_msg.data = float(accuracy)
            self.accuracy_pub.publish(accuracy_msg)

            results_msg = String()
            results_msg.data = json.dumps({
                'accuracy': accuracy,
                'classification_report': report,
                'confusion_matrix': conf_matrix.tolist(),
                'model_path': model_path
            })
            self.results_pub.publish(results_msg)

            status_msg.data = "TRAINING_COMPLETED"
            self.status_pub.publish(status_msg)

            self.get_logger().info(f'Breast Cancer training completed with accuracy: {accuracy:.4f}')

        except Exception as e:
            self.get_logger().error(f'Training failed: {str(e)}')
            status_msg.data = f"TRAINING_FAILED: {str(e)}"
            self.status_pub.publish(status_msg)

def main(args=None):
    rclpy.init(args=args)
    node = BreastCancerTrainingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
