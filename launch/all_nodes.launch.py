from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    package_dir = get_package_share_directory('ml_prediction_package')

    return LaunchDescription([
        # Include predictors
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                os.path.join(package_dir, 'launch', 'predictors.launch.py')
            ])
        ),

        # Include trainers
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                os.path.join(package_dir, 'launch', 'trainers.launch.py')
            ])
        ),

        # Include coordinator
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([
                os.path.join(package_dir, 'launch', 'coordinator.launch.py')
            ])
        ),
    ])