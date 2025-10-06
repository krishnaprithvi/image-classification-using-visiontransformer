# 🧠 Image Classification Using Vision Transformer (ViT)
This project explores image classification using the Vision Transformer (ViT) model architecture.
It demonstrates how Transformer-based models—originally designed for NLP—can be effectively applied to image recognition tasks.

The system is trained to classify medical images into specific categories (such as cataract detection) using both pretrained and custom-built models.


# 📘 Overview
The project implements and compares multiple deep learning approaches for image classification:
1. Vision Transformer (ViT) for transfer learning
2. Custom CNN and MLP architectures for comparison
3. Training from scratch and fine-tuning pretrained models
4. Tools for model evaluation, visualization, and result interpretation
The main focus is on understanding the role of attention mechanisms in visual feature learning.


# ✨ Key Features
1. 🧩 Transformer-Based Classification – Uses the ViT architecture for high-accuracy image classification
2. 🔁 Transfer Learning – Fine-tunes pretrained ViT models on the provided dataset.
3. 🧠 From Scratch Models – Includes CNN and MLP baselines for comparative study.
4. 📊 Visualization Tools – Generates training curves and attention maps for deeper insight.
5. ⚙️ Reusable Functions – Modular helper scripts (helper_functions.py) simplify experimentation.


# 📂 Project Structure
ImageClassificationUsingVisionTransformer/

├── AI.py                                 # Core script for Vision Transformer training

├── AIProject.py                          # End-to-end training and evaluation

├── helper_functions.py                   # Utility functions for data loading, metrics, etc.

├── train_using_pretrained_model_image_classifier.ipynb  # ViT fine-tuning notebook

├── image_classifier_from_scratch.ipynb   # Custom CNN model training

├── Multi-LayerPerceptron.ipynb           # MLP baseline

├── AI.ipynb / AIProjectCode.ipynb        # Main experimentation notebooks

├── Training/                             # Image dataset (e.g., cataract detection images)

└── README.md                             # Project documentation


# 🧰 Tools and Technologies
| **Component**           | **Technology Used**      |
| ----------------------- | ------------------------ |
| Language                | Python 3.8+              |
| Deep Learning Framework | PyTorch / torchvision    |
| Model Architecture      | Vision Transformer (ViT) |
| Visualization           | Matplotlib, Seaborn      |
| Data Handling           | NumPy, Pandas, Pillow    |
| Development             | Jupyter Notebook         |


# ⚙️ Setup Instructions
## Clone the repository
git clone https://github.com/<your-username>/ImageClassificationUsingVisionTransformer.git

cd ImageClassificationUsingVisionTransformer

## Install dependencies
pip install -r requirements.txt

## Prepare the dataset
Place your training and testing images under the respective directories (e.g., Training/, Testing/).

The dataset can be replaced with any custom image dataset.

## Run the notebook or script
python AI.py

or open and execute: train_using_pretrained_model_image_classifier.ipynb


# 🚀 How It Works
## Data Loading
The images are preprocessed (resized, normalized) and converted into tensor batches.

## Model Training
The Vision Transformer or CNN model is trained using classification loss and optimized via backpropagation.

## Evaluation
Model accuracy, loss, and confusion matrices are plotted to evaluate performance.

## Visualization
Attention heatmaps and training metrics are visualized to understand model learning behavior.


# 📈 Results
1. Pretrained Vision Transformer achieved higher accuracy and faster convergence than CNN and MLP baselines.
2. Training metrics, accuracy curves, and confusion matrices are available in the notebooks.


# 🧪 Experimentation Tips
1. Adjust learning rate, batch size, and optimizer in the config section.
2. Try different data augmentations to improve robustness.
3. Compare fine-tuned vs. from-scratch performance.


# 📘 License
This project is open for academic and research use.
Feel free to modify or extend it for experiments related to vision transformers, medical imaging, or transfer learning.