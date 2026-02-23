Project Overview

This project explores the impact of different optimizers and learning rate strategies on the performance of a Deep Neural Network (DNN) for a multi-class classification problem.

The notebook demonstrates:

Data preprocessing and encoding

PCA-based visualization

Model building using TensorFlow/Keras

Training with different optimizers (SGD, Momentum, RMSProp, Adam)

Learning rate scheduling

Performance comparison using training and validation loss curves

The goal of this project is to understand how optimizer selection and hyperparameter tuning affect convergence speed, training stability, and generalization performance.

Dataset

File: multiclass.csv

Type: Multi-class classification (3 classes)

One-hot encoding applied to categorical columns (Region and class)

Feature scaling performed using StandardScaler

Data split into train, validation, and test sets

Tech Stack

Python

NumPy

Pandas

Matplotlib

Seaborn

Scikit-learn

TensorFlow / Keras

Project Workflow

Data Preprocessing

Loaded dataset using Pandas

Applied one-hot encoding to categorical variables

Split data into training, validation, and test sets

Standardized numerical features

PCA Visualization

Reduced dimensionality to 2 components using PCA

Visualized class separability

Analyzed explained variance ratio

Deep Neural Network Architecture

Model Structure:

Dense (32) with ReLU activation

Dense (64) with ReLU activation

Dense (128) with ReLU activation

Dense (64) with ReLU activation

Dense (32) with ReLU activation

Output layer (3 neurons) with Softmax activation

Loss Function:
Categorical Crossentropy

Optimizers Compared

SGD (Vanilla)

Baseline optimizer

Slower convergence

More oscillation in loss

SGD with Momentum (0.9)

Faster convergence

Reduced oscillations

Improved stability

RMSProp

Adaptive learning rate

Handles varying gradient magnitudes effectively

Faster convergence than SGD

Adam (beta1 = 0.9, beta2 = 0.999)

Combines Momentum and RMSProp

Fast and stable convergence

Strong default optimizer for deep networks

Learning Rate Decay

Implemented a custom learning rate scheduler:

lr = lr / (1 + r0 * epoch)

Used Keras LearningRateScheduler callback to dynamically adjust learning rate across epochs.

Benefits:

Smoother convergence

Reduced overshooting

Improved validation performance

Observations

Vanilla SGD converges slowly compared to adaptive optimizers.

Momentum significantly accelerates convergence.

RMSProp stabilizes training further.

Adam achieves the fastest and smoothest convergence.

Learning rate decay improves generalization and reduces validation loss fluctuations.

Key Takeaways

Optimizer choice strongly influences convergence behavior.

Adaptive optimizers such as Adam and RMSProp generally outperform vanilla SGD.

Learning rate scheduling is crucial for stable and efficient training.

Proper preprocessing (encoding and scaling) is essential for DNN performance.

How to Run

Clone the repository:
git clone <your-repository-link>

Install dependencies:
pip install -r requirements.txt

Run the notebook:
jupyter notebook Optimizers_and_Hyperparams_in_deepNN.ipynb

Future Improvements

Add Early Stopping

Evaluate using Accuracy and F1-score

Perform systematic hyperparameter tuning

Add Dropout and Regularization

Compare with AdamW and Nadam
