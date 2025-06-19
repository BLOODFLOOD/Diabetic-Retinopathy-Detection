# 🩺 Diabetic Retinopathy Detection

A deep learning project to detect the severity of diabetic retinopathy using retinal fundus images. The model classifies images into 5 stages: **No DR**, **Mild**, **Moderate**, **Severe**, and **Proliferative DR**, using transfer learning with ResNet18.

---

## 📁 Project Structure

```
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
```

---

## 🔍 Dataset
link-https://www.kaggle.com/code/kushalkumar8906kumar/hiee-project/notebook
- **Directory**: `colored_images/` with subfolders for each DR category
- **Labels**: Provided in `train.csv`
- **Classes**:
  - `No_DR`
  - `Mild`
  - `Moderate`
  - `Severe`
  - `Proliferate_DR`

---

## 🧠 Model Details

- Framework: **TensorFlow / Keras**
- Architecture: **ResNet18** via transfer learning
- Classification Type: Multiclass (5 classes)
- Final Model Output: `retino_model.h5`
- Achieved Accuracy: **69%**

---

## 🚀 How to Run

### Step 1: Train the Model

````bash
jupyter notebook train.ipynb
````
Step 2: Start the Flask Web App
````bash
python main.py
````
Then go to http://127.0.0.1:5000/ in your browser.

✅ Features
Classifies 5 stages of diabetic retinopathy

Transfer learning with ResNet18

Web-based prediction interface using Flask

Real-world medical dataset with labeled fundus images

