from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='ml_prediction_package',
            executable='iris_predictor',
            name='iris_predictor_node',
            output='screen'
        ),
        Node(
            package='ml_prediction_package',
            executable='breast_cancer_predictor',
            name='breast_cancer_predictor_node',
            output='screen'
        ),
        Node(
            package='ml_prediction_package',
            executable='penguin_predictor',
            name='penguin_predictor_node',
            output='screen'
        ),
        Node(
            package='ml_prediction_package',
            executable='fruit_predictor',
            name='fruit_predictor_node',
            output='screen'
        ),
    ])