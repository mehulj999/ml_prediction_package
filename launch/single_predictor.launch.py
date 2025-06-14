from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    return LaunchDescription([
        # Arguments for selecting which predictor to run
        DeclareLaunchArgument(
            'predictor_name',
            default_value='iris_predictor',
            description='Name of the predictor node to launch',
            choices=['iris_predictor', 'breast_cancer_predictor',
                    'penguin_predictor', 'fruit_predictor']
        ),
        DeclareLaunchArgument(
            'node_name',
            default_value='',
            description='Custom node name (optional)'
        ),

        Node(
            package='ml_prediction_package',
            executable=LaunchConfiguration('predictor_name'),
            name=[LaunchConfiguration('predictor_name'), '_node'],
            output='screen'
        ),
    ])