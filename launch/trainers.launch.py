from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Get the path to the config file
    config_dir = os.path.join(get_package_share_directory('ml_prediction_package'), 'config')
    params_file = os.path.join(config_dir, 'training_params.yaml')

    return LaunchDescription([
        # Declare launch arguments
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('params_file', default_value=params_file),

        # Launch all training nodes WITHOUT the coordinator
        Node(
            package='ml_prediction_package',
            executable='iris_training_node',
            name='iris_training_node',
            parameters=[LaunchConfiguration('params_file')],
            output='screen'
        ),

        Node(
            package='ml_prediction_package',
            executable='breast_cancer_training_node',
            name='breast_cancer_training_node',
            parameters=[LaunchConfiguration('params_file')],
            output='screen'
        ),

        Node(
            package='ml_prediction_package',
            executable='penguin_training_node',
            name='penguin_training_node',
            parameters=[LaunchConfiguration('params_file')],
            output='screen'
        ),

        Node(
            package='ml_prediction_package',
            executable='fruit_training_node',
            name='fruit_training_node',
            parameters=[LaunchConfiguration('params_file')],
            output='screen'
        ),
    ])