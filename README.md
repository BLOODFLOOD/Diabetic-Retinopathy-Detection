🩺 Diabetic Retinopathy Detection
This deep learning project detects the severity of diabetic retinopathy from retinal fundus images. Using transfer learning with ResNet18, the model classifies images into five categories: No DR, Mild, Moderate, Severe, and Proliferative DR.

📁 Project Structure-

colored_images/
├── Mild/
├── Moderate/
├── No_DR/
├── Proliferate_DR/
└── Severe/

templates/
└── index.html

train.csv
train.ipynb
main.py
retino_model.h5 (Generated after training)

🔍 Dataset
Source: Folder colored_images/ with 5 subfolders for each class

Labels: Stored in train.csv

Classes:

No_DR

Mild

Moderate

Severe

Proliferate_DR

🧠 Model
Framework: TensorFlow / Keras

Architecture: ResNet18 (Transfer Learning)

Classification: 5-Class Multiclass

Output File: retino_model.h5

🚀 How to Run
Train the Model

Open and run all cells in train.ipynb:

This will generate and save the trained model as retino_model.h5.

Start the Web App

After training, run the Flask app:
python main.py
Then open http://127.0.0.1:5000/ in your browser to test predictions.

✅ Features
5-Class diabetic retinopathy detection

Achieved 69% accuracy

User-friendly web interface

ResNet18-based transfer learning




