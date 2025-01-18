# Depth Estimation Using ResNet50 and XGBoost
## Author
 - **Vishal Adithya.A**
## Overview
This project demonstrates a depth estimation XgBoost Regressor model that predicts the average depth of images provided using features extracted from a pre-trained ResNet50 model.The model was trained upon the **NYUv2 dataset** ([0jl/NYUv2](https://huggingface.co/datasets/0jl/NYUv2)). The trained model is saved using Python's `pickle` library for easy deployment and reuse.

### Loading the Model
The model is saved as `model.pkl` using `pickle`. You can load and use it as follows:

```python
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

features = extract_features("path/to/image.jpg") 
predicted_depth = model.predict([features])
print(predicted_depth[0])
```
**NOTE:** extract_features() is a predefined function in the original code which uses ResNet50 to extract features out of the image.

## Key Features
- **Model Architecture**:
  - Feature extraction: ResNet50 (pre-trained on ImageNet, with the top layers removed and global average pooling).
  - Regression: XGBoost, optimized for structured data prediction.
- **Training GPU**: NVIDIA RTX 4060 Ti, ensuring efficient computation.
- **Target**: Predict the average depth of images based on the depth maps from the dataset.

## Dataset
- Dataset: **NYUv2** ([0jl/NYUv2](https://huggingface.co/datasets/0jl/NYUv2))
- Format: The dataset includes RGB images and corresponding depth maps.
- Preprocessing:
  - Images were resized to 224x224 pixels to match the input requirements of ResNet50.
  - Depth maps were converted into single average depth values.

## Model Training
1. **Feature Extraction**:
   - ResNet50 was used to extract a fixed-length feature vector from each image.
   - Preprocessing: Images were normalized using the `preprocess_input` function from TensorFlow's ResNet50 module.
2. **Regression**:
   - XGBoost regressor was trained on the extracted features to predict average depth values.
   - Hyperparameters were tuned using cross-validation techniques for optimal performance.

## Results
- **R² Score**: 0.841
- Performance is reasonable for a first few implementation and can be further improved with additional tuning or by improving feature extraction methods.

## How to Use
### Requirements
1. Python 3.10+
2. Required libraries:
   - `numpy`
   - `pickle`
   - `xgboost`
   - `datasets`
   - `tensorflow`
   - `scikitlearn`

Install the dependencies using pip:
```bash
pip install numpy tensorflow xgboost datasets scikit-learn
```

### Training Pipeline
If you want to retrain the model, follow these steps:
 
1. Download the **NYUv2 dataset** from Hugging Face:
   ```python
   from datasets import load_dataset
   dataset = load_dataset("0jl/NYUv2")
   ```
2. Extract features using ResNet50:
   ```python

   model = ResNet50(weights="imagenet", include_top=False, pooling="avg")

   from PIL import Image
   def extract_features(image_path):
       image_array = preprocess_input(image_array)
       features = model.predict(image_array)
       return features.flatten()
   ```
3. Train the XGBoost regressor on the extracted features and save the model:
   ```python

   regressor = XGBRegressor()
   regressor.fit(X_train, y_train)

   with open("model.pkl", "wb") as f:
       pickle.dump(regressor, f)
   ```
**NOTE:** This pipeline has just the base fundamental code more additional parameter tunings and preprocessing steps were being conducted during the training of the original model.


## License
This project is licensed under the Apache License 2.0.

## Acknowledgments
- Hugging Face for hosting the NYUv2 dataset.
- NVIDIA RTX 4060 Ti for providing efficient GPU acceleration.
- TensorFlow and XGBoost for robust machine learning frameworks.
