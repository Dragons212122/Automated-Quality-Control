# O.O.F: Automated Quality Control 🏭

## 1. Refined Problem Statement
Manual quality inspection on manufacturing assembly lines is slow, inconsistent, and prone to human fatigue. This project automates the detection of surface defects on metal casting parts using a Convolutional Neural Network (CNN) to ensure reliable, high-speed quality control.

## 2. Research Questions
1. How accurately can a baseline CNN distinguish between defective and normal casting parts?
2. What data augmentation techniques best prevent the model from overfitting to the specific factory lighting of the training set?
3. Can the model achieve an inference speed fast enough for real-time assembly line deployment?

## 3. Project Structure
* `data/`: Raw and processed image datasets (not tracked by git).
* `notebooks/`: Jupyter notebooks for EDA and model development.
* `scripts/`: Python scripts for data preprocessing.
* `src/`: Core module containing CNN architectures.

## Setup Instructions
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run data exploration (see `notebooks/01_data_exploration.ipynb`).
