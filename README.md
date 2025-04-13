# 🕵️‍♂️ Synthetic Identity Fraud Demo

This interactive Streamlit app demonstrates how adversarial AI — specifically Generative Adversarial Networks (GANs) — can be used to simulate synthetic identity fraud and evaluate the resilience of fraud detection models.

## 🚀 What This App Does

- Generates synthetic "identities" using a simple GAN
- Visualises fake credit features such as **credit score** and **age**
- Evaluates a basic XGBoost fraud model against these synthetic profiles
- Shows how some synthetic identities can bypass fraud detection
- Optionally compares real vs synthetic feature distributions

## 🧠 Why It Matters

Fraudsters are increasingly using synthetic identities — blending real and fake information to build fake customer profiles that pass traditional checks. This app shows how adversarial AI can test the blind spots in fraud models.

---

## 🔧 How It Works

### 1. Train a GAN
- A simple PyTorch GAN is trained on Gaussian noise to generate 2D feature vectors representing synthetic identities.

### 2. Generate Identities
- The GAN outputs fake data mimicking features like **credit score** and **age**.

### 3. Train a Fraud Model
- A lightweight XGBoost binary classifier is trained on mock data to simulate fraud detection.

### 4. Evaluate Model
- Predictions are made on the synthetic identities.
- Fraud labels (`0 = non-fraud`, `1 = fraud`) are assigned.
- If synthetic identities are misclassified as non-fraud, a warning is shown.

### 5. Visualise the Risk
- Real vs synthetic distributions are shown via scatter plots.
- You can optionally toggle visibility using the sidebar controls.

---

## 🖥️ How to Run the App

1. Clone this repo or copy `untitled3.py`, `requirements.txt`, and this `README.md`.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
