import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
import torch.optim as optim

# ----------------------------
# GAN CONFIGURATION
# ----------------------------

class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim),
        )

    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

# ----------------------------
# FAKE DATA GENERATION
# ----------------------------

def train_gan(epochs=1000, latent_dim=10, output_dim=2):
    generator = Generator(latent_dim, output_dim)
    discriminator = Discriminator(output_dim)

    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=0.001)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.001)

    real_data = torch.tensor(np.random.normal(0, 1, (1000, output_dim)), dtype=torch.float32)

    for epoch in range(epochs):
        # Train Discriminator
        z = torch.randn(64, latent_dim)
        fake_data = generator(z)

        real_labels = torch.ones(64, 1)
        fake_labels = torch.zeros(64, 1)

        real_loss = criterion(discriminator(real_data[:64]), real_labels)
        fake_loss = criterion(discriminator(fake_data.detach()), fake_labels)
        d_loss = real_loss + fake_loss

        optimizer_d.zero_grad()
        d_loss.backward()
        optimizer_d.step()

        # Train Generator
        z = torch.randn(64, latent_dim)
        fake_data = generator(z)
        g_loss = criterion(discriminator(fake_data), real_labels)

        optimizer_g.zero_grad()
        g_loss.backward()
        optimizer_g.step()

    return generator

# ----------------------------
# XGBoost Fraud Model
# ----------------------------

def train_xgboost():
    X = np.random.randn(1000, 2)
    y = np.random.binomial(1, 0.1, 1000)
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X, y)
    return model, X, y

# ----------------------------
# Streamlit UI
# ----------------------------

st.title("Advai: Synthetic Identity Fraud Demo")

st.sidebar.header("Controls")
latent_dim = st.sidebar.slider("Latent Dim (GAN)", 2, 20, 10)
samples_to_generate = st.sidebar.slider("Synthetic IDs to Generate", 50, 500, 100)
evaluate_model = st.sidebar.checkbox("Run Evaluation on Synthetic Identities")
show_comparison = st.sidebar.checkbox("Show Real vs Synthetic Comparison")

# Train GAN and Generate Data
st.subheader("Generated Synthetic Identities")
generator = train_gan(latent_dim=latent_dim)
z = torch.randn(samples_to_generate, latent_dim)
generated_data = generator(z).detach().numpy()
df_gen = pd.DataFrame(generated_data, columns=["credit score", "age"])
st.dataframe(df_gen.head())

fig, ax = plt.subplots()
ax.scatter(df_gen["credit score"], df_gen["age"], alpha=0.6, label="Synthetic IDs", color='orange')
ax.set_title("GAN Generated Synthetic Identity Features")
ax.set_xlabel("Credit Score")
ax.set_ylabel("Age")
st.pyplot(fig)

# Optional: Compare to real data distribution
if show_comparison:
    X_real = np.random.randn(500, 2)
    df_real = pd.DataFrame(X_real, columns=["credit score", "age"])

    fig2, ax2 = plt.subplots()
    ax2.scatter(df_real["credit score"], df_real["age"], alpha=0.4, label="Real IDs", color='blue')
    ax2.scatter(df_gen["credit score"], df_gen["age"], alpha=0.6, label="Synthetic IDs", color='orange')
    ax2.legend()
    ax2.set_title("Real vs Synthetic Identity Feature Space")
    ax2.set_xlabel("Credit Score")
    ax2.set_ylabel("Age")
    st.pyplot(fig2)

# Model Evaluation on Synthetic IDs
if evaluate_model:
    st.subheader("XGBoost Model Evaluation")
    model, X_real, y_real = train_xgboost()
    preds_fake = model.predict(df_gen.values)

    fake_result = pd.DataFrame(df_gen)
    fake_result["Prediction"] = preds_fake
    st.write("Model predicted 'non-fraud' for synthetic IDs:")
    st.dataframe(fake_result[fake_result["Prediction"] == 0].head())

    st.write("⚠️ These synthetic identities bypass the fraud model.")

    y_pred_real = model.predict(X_real)
    report = classification_report(y_real, y_pred_real, output_dict=True)
    st.write("Model Report on Real Data:")
    st.json(report)

st.sidebar.markdown("---")
st.sidebar.markdown("👁️‍🗨️ *This tool simulates adversarial attacks on fraud models.*")
