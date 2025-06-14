from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    config_dir = os.path.join(get_package_share_directory('ml_prediction_package'), 'config')
    params_file = os.path.join(config_dir, 'training_params.yaml')

    return LaunchDescription([
        # Arguments for selecting which node to run
        DeclareLaunchArgument(
            'trainer_name',
            default_value='iris_training_node',
            description='Name of the training node to launch',
            choices=['iris_training_node', 'breast_cancer_training_node',
                    'penguin_training_node', 'fruit_training_node']
        ),
        DeclareLaunchArgument('params_file', default_value=params_file),

        Node(
            package='ml_prediction_package',
            executable=LaunchConfiguration('trainer_name'),
            name=LaunchConfiguration('trainer_name'),
            parameters=[LaunchConfiguration('params_file')],
            output='screen'
        ),
    ])