from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'ml_prediction_package'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),

        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'models'), glob('ml_prediction_package/models/*.pkl')),
         # Include config files
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        # Include data files - THIS IS CRUCIAL
        (os.path.join('share', package_name, 'data'), glob('data/*.csv')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mehul',
    maintainer_email='mehulj999@hotmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    # entry_points={
    #     'console_scripts': [
    #         'iris_predictor_node = ml_prediction_package.iris_predictor_node:main',
    #         'breast_cancer_predictor_node = ml_prediction_package.breast_cancer_predictor_node:main',
    #         'penguin_predictor_node = ml_prediction_package.penguin_predictor_node:main',
    #         'fruit_predictor_node = ml_prediction_package.fruit_predictor_node:main',
    #         'iris_training_node = ml_prediction_package.iris_training_node:main',
    #         'breast_cancer_training_node = ml_prediction_package.breast_cancer_training_node:main',
    #         'penguin_training_node = ml_prediction_package.penguin_training_node:main',
    #         'fruit_training_node = ml_prediction_package.fruit_training_node:main',
    #         'training_coordinator_node = ml_prediction_package.training_coordinator_node:main',
    #     ],
    entry_points={
    'console_scripts': [
        # Training nodes
        'iris_training_node = ml_prediction_package.iris_training_node:main',
        'breast_cancer_training_node = ml_prediction_package.breast_cancer_training_node:main',
        'penguin_training_node = ml_prediction_package.penguin_training_node:main',
        'fruit_training_node = ml_prediction_package.fruit_training_node:main',
        'training_coordinator_node = ml_prediction_package.training_coordinator_node:main',

        # Predictor nodes (your existing ones)
        'iris_predictor = ml_prediction_package.iris_predictor:main',
        'breast_cancer_predictor = ml_prediction_package.breast_cancer_predictor:main',
        'penguin_predictor = ml_prediction_package.penguin_predictor:main',
        'fruit_predictor = ml_prediction_package.fruit_predictor:main',
    ],
    },
)
