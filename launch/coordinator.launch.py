from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='ml_prediction_package',
            executable='training_coordinator_node',
            name='training_coordinator_node',
            output='screen'
        ),
    ])