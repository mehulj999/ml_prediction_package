#!/usr/bin/env python3

import joblib
import numpy as np
import os

# Load the model (adjust path as needed)
model_path = "ml_prediction_package/models/penguin_model.pkl"
model = joblib.load(model_path)

print("Model loaded successfully!")
print(f"Model type: {type(model)}")

# Check model attributes
if hasattr(model, 'feature_names_in_'):
    print(f"Expected features: {model.feature_names_in_}")
if hasattr(model, 'classes_'):
    print(f"Model classes: {model.classes_}")
if hasattr(model, 'n_features_in_'):
    print(f"Number of features expected: {model.n_features_in_}")

print("\n" + "="*50)
print("TESTING DIFFERENT PENGUIN EXAMPLES")
print("="*50)

# Test cases with different penguin characteristics
test_cases = [
    {
        "name": "Small Adelie-like",
        "features": [35.0, 19.0, 175.0, 3000.0, 2.0, 0.0]  # Small, high depth, short flipper, light
    },
    {
        "name": "Large Gentoo-like",
        "features": [50.0, 14.0, 220.0, 5500.0, 0.0, 1.0]  # Large, low depth, long flipper, heavy
    },
    {
        "name": "Medium Chinstrap-like",
        "features": [45.0, 18.0, 190.0, 3800.0, 1.0, 1.0]  # Medium size, medium depth
    },
    {
        "name": "Extreme Adelie",
        "features": [32.0, 21.0, 170.0, 2800.0, 2.0, 0.0]  # Very small, very high depth
    },
    {
        "name": "Extreme Gentoo",
        "features": [55.0, 13.0, 230.0, 6000.0, 0.0, 1.0]  # Very large, very low depth
    }
]

species_map = {0: 'Adelie', 1: 'Chinstrap', 2: 'Gentoo'}

for i, test_case in enumerate(test_cases):
    print(f"\nTest {i+1}: {test_case['name']}")
    print(f"Features: {test_case['features']}")

    # Make prediction
    features = np.array([test_case['features']])
    prediction = model.predict(features)[0]

    print(f"Raw prediction: {prediction}")
    print(f"Species: {species_map.get(prediction, f'Unknown_{prediction}')}")

    # Get probabilities if available
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(features)[0]
        print("Probabilities:")
        for j, prob in enumerate(probabilities):
            print(f"  {species_map.get(j, f'Class_{j}')}: {prob:.4f}")
        print(f"Confidence: {np.max(probabilities):.4f}")

    print("-" * 30)

print("\n" + "="*50)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*50)

# Try to get feature importance if available
if hasattr(model, 'feature_importances_'):
    print("Feature importances:")
    feature_names = ['culmen_length', 'culmen_depth', 'flipper_length', 'body_mass', 'island', 'sex']
    for name, importance in zip(feature_names, model.feature_importances_):
        print(f"  {name}: {importance:.4f}")
elif hasattr(model, 'coef_'):
    print("Model coefficients:")
    feature_names = ['culmen_length', 'culmen_depth', 'flipper_length', 'body_mass', 'island', 'sex']
    for name, coef in zip(feature_names, model.coef_[0]):
        print(f"  {name}: {coef:.4f}")
else:
    print("Feature importance not available for this model type")

print("\n" + "="*50)
print("RECOMMENDATIONS")
print("="*50)

print("1. Check if the model was trained with the correct feature order")
print("2. Verify the encoding used for categorical variables (island, sex)")
print("3. Check if feature scaling was applied during training")
print("4. Ensure the model file is the correct/latest version")
print("5. Try different test values to see if predictions vary")