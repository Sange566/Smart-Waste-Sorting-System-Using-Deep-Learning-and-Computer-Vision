# Smart Waste Sorting System Using Deep Learning and Computer Vision

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

## 🌍 Project Description

The Smart Waste Sorting System leverages deep learning and computer vision to automatically classify waste materials into six categories: **cardboard**, **glass**, **metal**, **paper**, **plastic**, and **trash**. Built with sustainability in mind, this system addresses the global waste management crisis by providing an intelligent, automated solution that can be deployed in recycling facilities, smart cities, and public waste management systems.

By utilizing the lightweight MobileNetV2 architecture with transfer learning, the system achieves **~95% validation accuracy** while remaining efficient enough for edge devices and resource-constrained environments. This project demonstrates how AI can support scalable, low-cost solutions for improving global waste management practices and promoting environmental sustainability.

## ✨ Features

- **Real-time Classification**: Live webcam-based waste classification with immediate feedback
- **High Accuracy**: Achieves ~95% validation accuracy across all waste categories
- **Confidence Filtering**: Only displays predictions above 90% confidence threshold for reliability
- **Lightweight Architecture**: Optimized for edge devices using MobileNetV2 (~2.6M parameters)
- **Data Privacy**: Processes only region of interest (ROI) without storing personal data
- **Robust Performance**: Handles various lighting conditions and object orientations
- **Scalable Deployment**: Suitable for Raspberry Pi and other embedded systems

## 🛠️ Technologies Used

- **Python 3.8+**: Primary programming language
- **TensorFlow/Keras**: Deep learning framework for model development
- **OpenCV**: Computer vision library for image processing and webcam integration
- **MobileNetV2**: Efficient CNN architecture for mobile and edge deployment
- **NumPy**: Numerical computing and array operations
- **Matplotlib & Seaborn**: Data visualization and performance analysis
- **Scikit-learn**: Model evaluation metrics and classification reports

## 🏗️ Model Architecture & Training

### Architecture Overview
The model combines MobileNetV2 as a feature extractor with a custom classification head:

```
Input (160×160×3) → MobileNetV2 (pretrained) → GlobalAveragePooling2D → 
BatchNormalization → Dropout(0.5) → Dense(256, ReLU) → 
BatchNormalization → Dropout(0.5) → Dense(6, Softmax)
```

### Training Strategy
The training process follows a two-phase approach:

**Phase 1: Transfer Learning**
- Freeze MobileNetV2 base model
- Train custom classifier head only
- Learning rate: 1e-4
- Epochs: 20

**Phase 2: Fine-tuning**
- Unfreeze all layers
- End-to-end training with reduced learning rate
- Learning rate: 1e-5
- Epochs: 10

### Key Hyperparameters
- **Batch Size**: 32
- **Input Resolution**: 160×160 pixels
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Dropout Rate**: 0.5
- **Data Augmentation**: Rotation, zoom, brightness, flipping, shifting

## 📊 Dataset Details

The dataset consists of **3,272 images** distributed across six waste categories:

| Category | Images | Percentage |
|----------|---------|------------|
| Cardboard | 713 | 21.8% |
| Glass | 751 | 22.9% |
| Metal | 660 | 20.2% |
| Paper | 844 | 25.8% |
| Plastic | 844 | 25.8% |
| Organic/Trash | 387 | 11.8% |

### Preprocessing & Balancing
- **Image Resizing**: All images resized to 160×160 pixels
- **Normalization**: Applied MobileNetV2's preprocess_input() function
- **Data Augmentation**: Applied 2x augmentation to minority classes (especially organic waste)
- **Train/Validation Split**: 80/20 ratio with stratified sampling
- **Quality Control**: Manual verification for labeling accuracy and visual clarity

## 🚀 Installation Instructions

### Prerequisites
- Python 3.8 or higher
- Webcam or camera device
- 4GB+ RAM recommended

### Setup Environment

1. **Clone the repository**
```bash
git clone https://github.com/sange566/Smart-Waste-Sorting-System-Using-Deep-Learning-and-Computer-Vision
.git
cd Smart-Waste-Sorting-System-Using-Deep-Learning-and-Computer-Vision

```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install tensorflow opencv-python numpy matplotlib seaborn scikit-learn
```

### Download Dataset
```bash
# Download from Kaggle or prepare your own dataset
# Ensure directory structure:
# dataset/
#   ├── cardboard/
#   ├── glass/
#   ├── metal/
#   ├── paper/
#   ├── plastic/
#   └── trash/
```

## 💻 Usage

### Training the Model
```bash
python Exercise3_code.py
```

### Real-time Waste Classification
```bash
python Webcam.py
```

### Using the Webcam Demo
1. Run the webcam script
2. Position waste item within the green bounding box
3. The system will display the predicted class and confidence percentage
4. Only predictions above 90% confidence are shown
5. Press 'q' to quit

### Sample Prediction Code
```python
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load model
model = load_model("best_model.h5")

# Classify image
def classify_waste(image_path):
    img = cv2.imread(image_path)
    img_resized = cv2.resize(img, (160, 160))
    img_processed = preprocess_input(img_resized)
    prediction = model.predict(np.expand_dims(img_processed, axis=0))
    class_id = np.argmax(prediction)
    confidence = prediction[0][class_id]
    
    classes = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
    return classes[class_id], confidence
```

## 📈 Results

### Model Performance
- **Final Validation Accuracy**: 95%
- **Model Size**: ~2.6M parameters (lightweight for edge deployment)
- **Training Time**: ~2 hours on GPU

### Classification Report
| Class | Precision | Recall | F1-Score |
|-------|-----------|---------|----------|
| Cardboard | 0.91 | 0.96 | 0.94 |
| Glass | 0.87 | 0.85 | 0.86 |
| Metal | 0.89 | 0.92 | 0.91 |
| Paper | 0.94 | 0.89 | 0.91 |
| Plastic | 0.78 | 0.71 | 0.74 |
| Trash | 0.93 | 0.87 | 0.90 |

### Key Insights
- **Strongest Performance**: Cardboard and trash categories (F1-score > 0.90)
- **Challenging Classification**: Plastic shows lowest recall (0.71), often confused with glass/metal
- **Visual Similarities**: Some confusion between plastic and glass due to transparency and texture
- **Overall Reliability**: High precision across all categories suitable for practical deployment

## 🔒 Ethical Considerations

### Data Privacy
- **No Personal Data Storage**: System processes only waste objects within ROI
- **Local Processing**: All predictions performed locally without data transmission
- **Minimal Data Collection**: No user images or personal information stored

### Bias and Fairness
- **Balanced Dataset**: Applied targeted augmentation to address class imbalances
- **Continuous Monitoring**: Regular evaluation to identify and mitigate biased predictions
- **Inclusive Design**: Tested across diverse lighting conditions and object orientations

### Accessibility and Sustainability
- **Low-Resource Deployment**: Optimized for edge devices and resource-constrained environments
- **Open Source**: Freely available for educational and research purposes
- **Environmental Impact**: Promotes recycling efficiency and reduces waste contamination

## 📸 Screenshots & Visualizations

The project includes comprehensive visualizations:
- Model architecture summary
- Training/validation accuracy curves
- Confusion matrix heatmap
- Classification report tables
- Real-time webcam interface screenshots

*Note: Add actual screenshots to `/images/` directory*

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments & References

### Datasets
- [Kaggle Waste Classification Dataset](https://www.kaggle.com/techsash/waste-classification-data)
- [OpenML Public Datasets](https://www.openml.org/)

### Key Papers & Libraries
- Howard, A. G., et al. (2017). "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications." arXiv:1704.04861
- [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
- [OpenCV Library](https://docs.opencv.org/)
- [Keras API Reference](https://keras.io/api/)

### Special Thanks
- University of the Western Cape - IFS315/IFS354 Course
- Lecturer: Ruchen Wyngaard
- Open-source community for tools and resources

## 📞 Contact

**Author**: Sangesonke Njameni 
**Email**: sangesonkenjameni5@gmail.com
**Institution**: University of the Western Cape  

For questions or collaborations, please open an issue or contact through GitHub.

---

*This project demonstrates the practical application of AI in environmental sustainability and waste management. Together, we can build smarter, cleaner cities through innovative technology solutions.*
