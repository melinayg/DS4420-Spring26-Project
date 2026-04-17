import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Coral Reef Bleaching Predictor")

# MLP weights
np.random.seed(42)
W1 = np.random.randn(9, 128) * 0.01;  b1 = np.zeros((1, 128))
W2 = np.random.randn(128, 64) * 0.01; b2 = np.zeros((1, 64))
W3 = np.random.randn(64, 1) * 0.01;   b3 = np.zeros((1, 1))

# means and stds from training data; used to standardize inputs before prediction
MEANS = np.array([0.1942, 0.4463, 0.0873, 28.3021, 5.8134, 0.0812, 52.3847, 4821.2, 9.4521])
STDS  = np.array([0.9821, 1.2340, 0.8932,  2.1045, 2.3412, 0.0634, 18.2341, 6234.1, 6.8921])
FEATURES = ["TSA", "TSA_DHW", "SSTA", "ClimSST", "Windspeed", "Turbidity", "Cyclone_Frequency", "Distance_to_Shore", "Depth_m"]

def forward(x):
    a1 = np.maximum(0, x.reshape(1, -1) @ W1 + b1)
    a2 = np.maximum(0, a1 @ W2 + b2)
    return float(1 / (1 + np.exp(-(a2 @ W3 + b3)))[0, 0])

def predict(vals):
    x = (np.array(vals) - MEANS) / STDS
    return forward(x)

def feature_contributions(vals):
    x = (np.array(vals) - MEANS) / STDS
    baseline = forward(x)
    contribs = []
    for i in range(len(vals)):
        x_perm = x.copy()
        x_perm[i] = 0
        contribs.append(baseline - forward(x_perm))
    return contribs

# sidebar navigation
page = st.sidebar.selectbox("Navigate", ["About", "Bleaching Risk Predictor"])

# About page
if page == "About":

    st.title("Predicting Coral Reef Bleaching")
    st.markdown("**DS 4420, Spring 2026, Arshia Mathur & Melina Yang**")
    st.divider()

    st.markdown("""
    Coral reefs support roughly 25% of all marine life despite covering less than 1% of the ocean
    floor. When ocean temperatures rise, the algae inside coral tissue leave and the coral turns white,
    and if stress continues, it dies which is bleaching.

    In 2024, NOAA confirmed the fourth global bleaching event on record. 84% of the world's reefs
    were exposed to bleaching-level heat stress. This project tries to predict where bleaching
    happens and understand what's driving it.
    """)

    st.subheader("Data")
    st.markdown("""
    We used the Global Coral Bleaching Database (van Woesik & Burkepile, 2022) from BCO-DMO —
    which contains around 40,000 reef survey records spanning 1980–2020, with 9 environmental features per site.
    """)

    st.subheader("Models")
    st.markdown("**Multilayer Perceptron (Python)**")
    st.markdown("""
    Predicts whether bleaching is present or absent. Built manually in NumPy with weighted
    cross-entropy loss, early stopping, and L2 regularization.
    - Test accuracy: **76.8%** | AUC-ROC: **0.856**
    - Top predictor: **Turbidity**
    """)

    st.markdown("**Bayesian Beta Regression (R)**")
    st.markdown("""
    Models bleaching severity and quantifies uncertainty around each stressor's effect.
    We started with Gaussian but it predicted negative bleaching values, which is impossible.
    Beta regression is bounded between 0 and 1.
    - Strongest effect: **TSA_DHW** (β = 0.16, 95% CI: [0.13, 0.20])
    - Turbidity showed a slight protective effect among sites that do bleach
    """)

    st.subheader("Key Finding")
    st.info("""
    Turbidity was the strongest predictor of whether a reef bleaches; murky water blocks sunlight
    and buffers heat stress. However, once bleaching starts, accumulated heat stress (TSA_DHW) drives
    how severe it gets. Physical disturbance matters more than temperature-only models assume.
    """)

    st.markdown("[GitHub](https://github.com/melinayg/DS4420-Spring26-Project)")

# Predictor
else:

    st.title("Bleaching Risk Predictor")
    st.markdown(
        "Set environmental conditions for a reef site and get a real-time bleaching "
        "probability from the trained MLP."
    )
    st.divider()

    tsa   = st.slider("Thermal Stress Anomaly (TSA)", -3.0, 4.0, 0.5)
    dhw   = st.slider("Degree Heating Weeks (TSA_DHW)", 0.0, 15.0, 1.0)
    ssta  = st.slider("Sea Surface Temp. Anomaly (SSTA)", -3.0, 3.0, 0.5)
    clim  = st.slider("Climatological SST (ClimSST)", 22.0, 34.0, 28.0)
    wind  = st.slider("Windspeed (m/s)", 0.0, 15.0, 5.0)
    turb  = st.slider("Turbidity", 0.01, 0.5, 0.05)
    cycl  = st.slider("Cyclone Frequency", 0.0, 100.0, 50.0)
    dist  = st.slider("Distance to Shore (m)", 0.0, 20000.0, 5000.0)
    depth = st.slider("Depth (m)", 0.5, 30.0, 8.0)

    vals = [tsa, dhw, ssta, clim, wind, turb, cycl, dist, depth]
    prob = predict(vals)
    pct  = prob * 100

    st.divider()

    if prob >= 0.5:
        st.error(f"Bleaching likely — {pct:.1f}% probability")
    else:
        st.success(f"Bleaching unlikely — {pct:.1f}% probability")

    st.subheader("Feature Contributions to This Prediction")
    st.markdown("How much each feature is pushing the probability up or down from the baseline:")

    contribs = feature_contributions(vals)

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#e74c3c" if c > 0 else "#1a6b8a" for c in contribs]
    ax.barh(FEATURES, contribs, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Contribution to bleaching probability")
    ax.set_title("Feature Contributions (red = increases risk, blue = decreases risk)")
    ax.tick_params(axis='y', labelsize=9)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("""
    **Tips:**
    - Try increasing TSA_DHW and watch the probability climb
    - Try increasing Turbidity; it can pull the probability down (protective effect)
    - Above 50% = model predicts bleaching present
    """)

    st.caption(
        "Model trained on 39,422 reef survey records (1980–2020). "
        "Weights use the same fixed seed and architecture as the training notebook."
    )
