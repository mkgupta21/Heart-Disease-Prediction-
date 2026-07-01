# ❤️ CardioPredict - Heart Disease Prediction Dashboard

CardioPredict is a high-fidelity data application that uses a trained Support Vector Machine (SVM) classifier to predict the probability of heart disease in patients based on 13 clinical parameters. The project includes the full model training notebook, processed datasets, and a styled Streamlit web application.

---

## 📂 Repository Structure

The project has been reorganized into a clean, professional structure:

```
Heart-Disease-Prediction/
├── data/
│   ├── labels.csv                      # Target labels (presence of heart disease)
│   └── values.csv                      # Clinical patient features
├── models/
│   ├── heart_model.pkl                 # Trained Support Vector Machine model
│   └── scaler.pkl                      # Fit StandardScaler for continuous features
├── notebooks/
│   └── Heart_Disease_Prediction.ipynb  # Jupyter notebook containing EDA, training, & evaluation
├── .gitignore                          # Excludes cache files and virtual environments
├── app.py                              # Upgraded Streamlit diagnostic web application
├── requirements.txt                    # Python library requirements
└── README.md                           # Comprehensive project documentation
```

---

## 🧬 Dataset Features

The model processes 13 patient features to make its diagnostic prediction:

### Continuous Features (Standard Scaled)
- **Age**: Patient's age (years)
- **Resting Blood Pressure**: Resting blood pressure (mm Hg) on admission to the hospital
- **Serum Cholesterol**: Serum cholesterol level (mg/dl)
- **Max Heart Rate**: Maximum heart rate achieved (bpm)
- **Oldpeak**: ST depression induced by exercise relative to rest (log-transformed: `np.log1p`)

### Categorical & Binary Features
- **Sex**: Biological sex (Male = 1, Female = 0)
- **Chest Pain Type**: (Type 1: Typical Angina, Type 2: Atypical Angina, Type 3: Non-anginal pain, Type 4: Asymptomatic)
- **Exercise Induced Angina**: Angina (chest pain) induced by exercise (Yes = 1, No = 0)
- **Slope of ST Segment**: Slope of the peak exercise ST segment (1: Upsloping, 2: Flat, 3: Downsloping)
- **Number of Major Vessels**: Number of major blood vessels (0-3) colored by fluoroscopy
- **Fasting Blood Sugar**: Fasting blood sugar > 120 mg/dl (Yes = 1, No = 0)
- **Resting EKG Results**: (0: Normal, 1: ST-T wave abnormality, 2: Left ventricular hypertrophy)
- **Thal**: Thalassemia blood condition (one-hot encoded: Normal, Fixed Defect, Reversible Defect)

---

## 🧠 Machine Learning Pipeline & Results

The diagnostic classifier is built using a Support Vector Machine (SVM) pipeline:
1. **Preprocessing**: Handled outlier capping on continuous fields, applied a log transformation (`np.log1p`) on the right-skewed `oldpeak` column, and one-hot encoded `thal`.
2. **Feature Scaling**: Used a `StandardScaler` fitted on continuous features to prevent distance bias in SVM.
3. **Model Selection**: Evaluated Logistic Regression, Decision Trees, SVM, Random Forests, Gradient Boosting, and XGBoost.
4. **Hyperparameter Tuning**: Tuned the SVM hyperparameters using Randomized Search Cross-Validation (5-fold) optimizing for ROC AUC.
5. **Results**:
   - **Validation Accuracy**: `~83.3%`
   - **ROC AUC Score**: `~88.9%`

---

## 💻 Streamlit Web Application

The diagnostic application (`app.py`) provides an interactive interface built with a custom **Zinc CSS Design System** supporting:
- **Light / Dark Mode**: Instantly switch between themes with a clean toggle button in the header.
- **Diagnostic Panel**: Sliders and dropdowns for all 13 features, with advanced clinical parameters grouped in an expander with sensible baseline defaults.
- **Robust Inference**: Preprocesses user inputs correctly, applying the log transformation to `oldpeak` before scaling and predicting.
- **Interactive EDA Explorer**: Allows full interactive visualization of the 180 patient dataset using Plotly scatter, box, and histogram charts.

---

## 🚀 Getting Started

Follow these steps to run the diagnostic dashboard locally:

### Prerequisites
- Python 3.10+ installed

### Installation & Launch

1. **Clone the repository**:
   ```bash
   git clone https://github.com/mkgupta21/Heart-Disease-Prediction-.git
   cd Heart-Disease-Prediction-
   ```

2. **Set up a virtual environment**:
   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment**:
   - **Windows (PowerShell)**:
     ```powershell
     .venv\Scripts\Activate.ps1
     ```
   - **Windows (CMD)**:
     ```cmd
     .venv\Scripts\activate.bat
     ```
   - **macOS / Linux**:
     ```bash
     source .venv/bin/activate
     ```

4. **Install requirements**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Launch the web application**:
   ```bash
   streamlit run app.py
   ```