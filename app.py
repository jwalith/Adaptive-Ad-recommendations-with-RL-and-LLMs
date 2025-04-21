# Streamlit App with Real Criteo Data Integration

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ========== Section 1: Setup ==========
st.set_page_config(page_title="LinUCB Ad Recommender", layout="centered")
st.title("📢 LinUCB Ad Recommender (Real Criteo Data + LLM Embeddings)")
st.markdown("""
This dashboard shows a LinUCB contextual bandit model trained on real Criteo data,
enhanced with LLM embeddings from MiniLM for ad context enrichment.
""")

# Load preprocessed data (simulate loading previously computed)
@st.cache_data
def load_data():
    df = pd.read_csv("C:/Users/jwali/Downloads/Kaggle/RL/train.txt", sep="\t", header=None, nrows=10000)
    columns = ['label'] + [f'I{i}' for i in range(1, 14)] + [f'C{i}' for i in range(1, 27)]
    df.columns = columns
    df = df.dropna(subset=['label']).reset_index(drop=True)

    # Preprocess
    num_cols = ['I1', 'I2', 'I3']
    cat_cols = ['C1', 'C2']
    df[num_cols] = df[num_cols].fillna(0)
    df[cat_cols] = df[cat_cols].fillna("missing")

    scaler = StandardScaler()
    X_num = scaler.fit_transform(df[num_cols])

    encoders = {col: LabelEncoder().fit(df[col]) for col in cat_cols}
    X_cat = np.stack([encoders[col].transform(df[col]) for col in cat_cols], axis=1)

    X = np.concatenate([X_num, X_cat], axis=1)
    y = df['label'].astype(int).values

    return df, X, y, X_cat

df, X, y, X_cat = load_data()

# Simulate ad text mapping from C1 values
ad_texts = {
    0: "Discount running shoes for fitness lovers",
    1: "Zero-fee credit card with cashback rewards",
    2: "Next-gen gaming console with 4K graphics"
}

@st.cache_data
def load_embeddings():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return {k: model.encode([v])[0] for k, v in ad_texts.items()}

ad_embeddings = load_embeddings()
embedding_dim = next(iter(ad_embeddings.values())).shape[0]

# ========== LinUCB Class ==========
class LinUCB:
    def __init__(self, n_arms, n_features, alpha=0.5):
        self.alpha = alpha
        self.n_arms = n_arms
        self.n_features = n_features
        self.A = [np.identity(n_features) for _ in range(n_arms)]
        self.b = [np.zeros(n_features) for _ in range(n_arms)]

    def predict(self, x_contexts):
        p = np.zeros(self.n_arms)
        for a in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            x = x_contexts[a]
            p[a] = theta @ x + self.alpha * np.sqrt(x.T @ A_inv @ x)
        return np.argmax(p), p

    def update(self, chosen_arm, x, reward):
        self.A[chosen_arm] += np.outer(x, x)
        self.b[chosen_arm] += reward * x

# ========== Training with Real Data ==========
@st.cache_data
def train_with_criteo(X, y, X_cat):
    arms = 3
    context_dim = X.shape[1] + embedding_dim
    agent = LinUCB(n_arms=arms, n_features=context_dim, alpha=0.5)
    rewards = []

    for i in range(len(X)):
        x = X[i]
        true_arm = X_cat[i][0] % 3
        contexts = [np.concatenate([x, ad_embeddings[a]]) for a in range(arms)]
        chosen_arm, _ = agent.predict(contexts)
        reward = 1 if chosen_arm == true_arm and y[i] == 1 else 0
        agent.update(chosen_arm, contexts[chosen_arm], reward)
        rewards.append(reward)

    ctr_curve = np.cumsum(rewards) / (np.arange(len(rewards)) + 1)
    return agent, ctr_curve

agent, ctr_curve = train_with_criteo(X, y, X_cat)

# ========== Visualization ==========
st.subheader("📈 CTR Over Time on Criteo Data")
fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(ctr_curve)
ax.set_title("Cumulative CTR - LinUCB + MiniLM on Criteo")
ax.set_xlabel("Rounds")
ax.set_ylabel("CTR")
st.pyplot(fig)
st.metric("Final CTR", f"{ctr_curve[-1]:.3f}")

# ========== Live Prediction ==========
st.subheader("🎮 Live Ad Recommendation")
i = st.slider("Choose a row from dataset (0–9999)", 0, 9999, 0)
x = X[i]
true_arm = X_cat[i][0] % 3
contexts = [np.concatenate([x, ad_embeddings[a]]) for a in range(3)]
chosen_arm, scores = agent.predict(contexts)

st.write(f"**Predicted Ad:** #{chosen_arm}")
st.write(f"**Ad Text:** {ad_texts[chosen_arm]}")
st.write(f"**True Click Label:** {y[i]}")

st.bar_chart(scores)
st.caption("Built with ❤️ using real Criteo data, LinUCB, MiniLM, and Streamlit.")
