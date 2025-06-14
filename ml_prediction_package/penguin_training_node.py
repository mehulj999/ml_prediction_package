#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib
import json
import os
from ament_index_python.packages import get_package_share_directory

class PenguinTrainingNode(Node):
    def __init__(self):
        super().__init__('penguin_training_node')

        # Publishers
        self.status_pub = self.create_publisher(String, '/training/penguin/status', 10)
        self.accuracy_pub = self.create_publisher(Float32, '/training/penguin/accuracy', 10)
        self.results_pub = self.create_publisher(String, '/training/penguin/results', 10)

        # Subscribers
        self.train_sub = self.create_subscription(String, '/training/penguin/start', self.train_callback, 10)

        # Parameters
        self.declare_parameter('data_path', 'data/Penguin.csv')
        self.declare_parameter('model_output_path', 'models/penguin_svm_linear.pkl')

        #  # Parameters with default paths relative to package
        # self.declare_parameter('data_path', 'data/Iris.csv')
        # self.declare_parameter('model_output_path', 'models/iris_voting_classifier.pkl')

        self.get_logger().info('Penguin Training Node initialized')

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
        self.get_logger().info('Starting Penguin model training...')

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

            # Load and preprocess data
            df = pd.read_csv(data_path)
            df.dropna(inplace=True)

            # Encode categorical columns
            le_species = LabelEncoder()
            df['species_encoded'] = le_species.fit_transform(df['species'])

            le_sex = LabelEncoder()
            df['sex_encoded'] = le_sex.fit_transform(df['sex'])

            le_island = LabelEncoder()
            df['island_encoded'] = le_island.fit_transform(df['island'])

            # Define features
            features = ['culmen_length_mm', 'culmen_depth_mm', 'flipper_length_mm', 'body_mass_g',
                       'sex_encoded', 'island_encoded']
            X = df[features]
            y = df['species_encoded']

            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

            # Train SVM Linear Classifier
            model = SVC(kernel='linear', random_state=42)
            model.fit(X_train, y_train)

            # Evaluate
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, target_names=le_species.classes_, output_dict=True)

            # Save model
            joblib.dump(model, model_path)

            # Publish results
            accuracy_msg = Float32()
            accuracy_msg.data = float(accuracy)
            self.accuracy_pub.publish(accuracy_msg)

            results_msg = String()
            results_msg.data = json.dumps({
                'accuracy': accuracy,
                'classification_report': report,
                'model_path': model_path
            })
            self.results_pub.publish(results_msg)

            status_msg.data = "TRAINING_COMPLETED"
            self.status_pub.publish(status_msg)

            self.get_logger().info(f'Penguin training completed with accuracy: {accuracy:.4f}')

        except Exception as e:
            self.get_logger().error(f'Training failed: {str(e)}')
            status_msg.data = f"TRAINING_FAILED: {str(e)}"
            self.status_pub.publish(status_msg)

def main(args=None):
    rclpy.init(args=args)
    node = PenguinTrainingNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()