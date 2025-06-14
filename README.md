# ML Prediction Package

A ROS2 package for machine learning prediction and analysis featuring 4 different datasets and models for vehicular technology applications.

## Overview

This package provides both training and prediction capabilities for four different machine learning models:
- **Iris Classification** - Flower species classification
- **Breast Cancer Classification** - Medical diagnosis prediction
- **Penguin Classification** - Species identification
- **Fruit Classification** - Fruit type recognition

Each model includes dedicated training and prediction nodes that communicate via ROS2 topics and services.

## Package Structure

```
ml_prediction_package/
├── ml_prediction_package/
│   ├── models/          # Pre-trained model files (.pkl)
│   ├── *_training_node.py    # Training nodes for each model
│   ├── *_predictor.py        # Prediction nodes for each model
│   └── training_coordinator_node.py
├── data/                # Training datasets (.csv)
├── config/              # Configuration files (.yaml)
├── launch/              # Launch files (.launch.py)
├── setup.py
└── package.xml
```

## Prerequisites

- ROS2 (Humble/Iron/Rolling)
- Python 3.8+
- Required Python packages: scikit-learn, pandas, numpy

## Installation

1. Clone the repository into your ROS2 workspace:
```bash
cd ~/ros2_ws/src
git clone https://github.com/mehulj999/ml_prediction_package.git
```

2. Install dependencies:
```bash
cd ~/ros2_ws
rosdep install --from-paths src --ignore-src -r -y
```

3. Build the package:
```bash
colcon build --packages-select ml_prediction_package
source install/setup.bash
```

## Usage

### Training Nodes

Train individual models using the dedicated training nodes:

#### 1. Iris Model Training
```bash
ros2 run ml_prediction_package iris_training_node
```

#### 2. Breast Cancer Model Training
```bash
ros2 run ml_prediction_package breast_cancer_training_node
```

#### 3. Penguin Model Training
```bash
ros2 run ml_prediction_package penguin_training_node
```

#### 4. Fruit Model Training
```bash
ros2 run ml_prediction_package fruit_training_node
```

#### Training Coordinator (Train All Models)
```bash
ros2 run ml_prediction_package training_coordinator_node
```

### Prediction Nodes

Run prediction services for each trained model:

#### 1. Iris Predictor
```bash
ros2 run ml_prediction_package iris_predictor
```

#### 2. Breast Cancer Predictor
```bash
ros2 run ml_prediction_package breast_cancer_predictor
```

#### 3. Penguin Predictor
```bash
ros2 run ml_prediction_package penguin_predictor
```

#### 4. Fruit Predictor
```bash
ros2 run ml_prediction_package fruit_predictor
```

## Testing the Models

### Method 1: Using ROS2 Services (Recommended)

After starting a predictor node, you can test it using ROS2 service calls:

```bash
# Example for iris prediction
ros2 service call /iris_predict your_service_type "data: [5.1, 3.5, 1.4, 0.2]"
```

### Method 2: Using ROS2 Topics

Publishers and subscribers can be used to send data and receive predictions:

```bash
# List available topics
ros2 topic list

# Send test data (adjust topic name and message type accordingly)
ros2 topic pub /prediction_input std_msgs/Float32MultiArray "data: [5.1, 3.5, 1.4, 0.2]"

# Monitor predictions
ros2 topic echo /prediction_output
```

## Launch Files

Use launch files to start multiple nodes simultaneously:

```bash
# Train all models
ros2 launch ml_prediction_package train_all_models.launch.py

# Start all predictors
ros2 launch ml_prediction_package start_all_predictors.launch.py
```

## Configuration

Model parameters and training configurations can be modified in the `config/` directory. Each model may have its own YAML configuration file for hyperparameters and data preprocessing settings.

## Data

The package includes CSV datasets in the `data/` directory:
- `iris.csv` - Iris flower measurements
- `breast_cancer.csv` - Breast cancer diagnostic data
- `penguin.csv` - Palmer penguins data
- `fruit.csv` - Fruit classification data

## Model Files

Trained models are saved as pickle files in the `models/` directory and automatically loaded by prediction nodes.

## Development

### Adding New Models

1. Create training node: `new_model_training_node.py`
2. Create predictor node: `new_model_predictor.py`
3. Add dataset to `data/` directory
4. Update `setup.py` entry points
5. Add configuration file if needed

### Debugging

Enable ROS2 logging to debug node behavior:
```bash
ros2 run ml_prediction_package iris_predictor --ros-args --log-level DEBUG
```

## Troubleshooting

**Model not found errors**: Ensure training has been completed before running predictors.

**Import errors**: Verify all Python dependencies are installed:
```bash
pip install scikit-learn pandas numpy
```

**ROS2 communication issues**: Check that nodes are properly advertising/subscribing to topics:
```bash
ros2 node list
ros2 topic list
ros2 service list
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

TODO: License declaration

## Maintainer

**Mehul** - mehulj999@hotmail.com
