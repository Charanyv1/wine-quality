"""Streamlit front-end for the wine quality deep learning model.

Run with:  streamlit run app.py
The model must be trained first:  python train_model.py
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, 'artifacts')
MODEL_PATH = os.path.join(ARTIFACTS_DIR, 'wine_quality_dnn.keras')
SCALER_PATH = os.path.join(ARTIFACTS_DIR, 'scaler.pkl')
METRICS_PATH = os.path.join(ARTIFACTS_DIR, 'metrics.json')
LOSS_PLOT_PATH = os.path.join(ARTIFACTS_DIR, 'loss_plot.png')
ACCURACY_PLOT_PATH = os.path.join(ARTIFACTS_DIR, 'accuracy_plot.png')
CONFUSION_MATRIX_PATH = os.path.join(ARTIFACTS_DIR, 'confusion_matrix.png')

# (column name in the training data, label, help text, default, min, max, step, decimals)
FEATURES = [
    ('fixed acidity', 'Fixed Acidity', 'Tartaric acid (g/dm³). Typical range 4.6 – 15.9.', 8.3, 0.0, 20.0, 0.1, '%.2f'),
    ('volatile acidity', 'Volatile Acidity', 'Acetic acid (g/dm³). High values taste like vinegar.', 0.53, 0.0, 2.0, 0.01, '%.3f'),
    ('citric acid', 'Citric Acid', 'Adds freshness (g/dm³). Typical range 0.0 – 1.0.', 0.27, 0.0, 2.0, 0.01, '%.3f'),
    ('residual sugar', 'Residual Sugar', 'Sugar left after fermentation (g/dm³).', 2.5, 0.0, 20.0, 0.1, '%.2f'),
    ('chlorides', 'Chlorides', 'Salt content (g/dm³). Typical range 0.01 – 0.6.', 0.087, 0.0, 1.0, 0.001, '%.3f'),
    ('free sulfur dioxide', 'Free Sulfur Dioxide', 'Free SO₂ (mg/dm³). Prevents microbial growth.', 15.0, 0.0, 100.0, 1.0, '%.1f'),
    ('total sulfur dioxide', 'Total Sulfur Dioxide', 'Total SO₂ (mg/dm³). Typical range 6 – 289.', 46.0, 0.0, 400.0, 1.0, '%.1f'),
    ('density', 'Density', 'Density (g/cm³). Close to water, 0.990 – 1.004.', 0.9967, 0.9, 1.1, 0.0001, '%.4f'),
    ('pH', 'pH', 'Acidity scale. Most red wines sit between 2.7 and 4.0.', 3.31, 0.0, 14.0, 0.01, '%.2f'),
    ('sulphates', 'Sulphates', 'Potassium sulphate (g/dm³). Antioxidant additive.', 0.66, 0.0, 3.0, 0.01, '%.2f'),
    ('alcohol', 'Alcohol', 'Alcohol by volume (%). Typical range 8.4 – 14.9.', 10.4, 0.0, 20.0, 0.1, '%.2f'),
]

st.set_page_config(page_title='Wine Quality Prediction Using Deep Learning', page_icon='🍷', layout='wide')


@st.cache_resource(show_spinner='Loading the trained deep learning model…')
def load_artifacts():
    """Load the Keras model, scaler and metrics once per server process."""
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    with open(METRICS_PATH) as f:
        metrics = json.load(f)
    return model, scaler, metrics


st.title('🍷 Wine Quality Prediction Using Deep Learning')
st.caption(
    'A Keras deep neural network (128 → 64 → 32 → 1) trained on the UCI red wine dataset. '
    'A wine counts as good quality when its sensory score is 7 or higher.'
)

if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH) and os.path.exists(METRICS_PATH)):
    st.error('Trained model not found. Run `python train_model.py` first to create the artifacts/ directory.')
    st.stop()

model, scaler, metrics = load_artifacts()

st.subheader('Chemical properties')
values = {}
left, right = st.columns(2)
for index, (column, label, help_text, default, min_value, max_value, step, fmt) in enumerate(FEATURES):
    target = left if index % 2 == 0 else right
    values[column] = target.number_input(
        label,
        min_value=min_value,
        max_value=max_value,
        value=default,
        step=step,
        format=fmt,
        help=help_text,
    )

st.divider()

if st.button('Predict Wine Quality', type='primary', use_container_width=True):
    invalid = [label for column, label, *_ in FEATURES if values[column] < 0 or not np.isfinite(values[column])]
    if invalid:
        st.error(f'These fields need a finite, non-negative value: {", ".join(invalid)}.')
    elif all(values[column] == 0 for column, *_ in FEATURES):
        st.warning('All inputs are zero — enter the measurements of a real wine sample.')
    else:
        # Arrange the inputs in the exact order the model was trained on, scale them
        # with the fitted StandardScaler, then run a forward pass through the network.
        feature_order = metrics['feature_order']
        ordered = pd.DataFrame([[values[column] for column in feature_order]], columns=feature_order)
        scaled = scaler.transform(ordered)
        probability = float(model.predict(scaled, verbose=0).ravel()[0])
        is_good = probability >= 0.5
        confidence = probability if is_good else 1.0 - probability

        st.subheader('Prediction result')
        result, gauge = st.columns([2, 1])
        with result:
            if is_good:
                st.success('### 🍷 Good Quality Wine')
            else:
                st.warning('### ⚠️ Not a Good Quality Wine')
            st.metric('Model confidence', f'{confidence:.1%}')
            st.caption(f'P(good quality) = {probability:.4f}; the decision threshold is 0.50.')
        with gauge:
            st.progress(probability, text=f'P(good quality) {probability:.1%}')

with st.expander('About the Deep Learning Model'):
    test_metrics = metrics['test_metrics']
    st.markdown('**Architecture**')
    st.code('\n'.join(metrics['architecture']), language='text')

    col_a, col_b = st.columns(2)
    col_a.markdown(
        f"""
- **Hidden layers:** {metrics['hidden_layers']}
- **Trainable structure:** {metrics['total_parameters']:,} parameters
- **Activations:** ReLU (hidden) — non-saturating, keeps gradients flowing; Sigmoid (output) — maps the logit to a probability
- **Loss function:** {metrics['loss_function']} — the negative log-likelihood of a binary target
"""
    )
    col_b.markdown(
        f"""
- **Optimizer:** {metrics['optimizer']} (learning rate {metrics['learning_rate']})
- **Batch size:** {metrics['batch_size']}
- **Epochs completed:** {metrics['epochs_completed']} of a maximum {metrics['max_epochs']}
- **Early stopping:** {metrics['early_stopping']}
- **Split:** {metrics['dataset']['train']} train / {metrics['dataset']['validation']} validation / {metrics['dataset']['test']} test
"""
    )

    st.markdown('**Measured performance on the held-out test split**')
    m1, m2, m3, m4 = st.columns(4)
    m1.metric('Accuracy', f"{test_metrics['accuracy']:.3f}")
    m2.metric('Precision', f"{test_metrics['precision']:.3f}")
    m3.metric('Recall', f"{test_metrics['recall']:.3f}")
    m4.metric('F1-score', f"{test_metrics['f1']:.3f}")
    st.caption(f"ROC AUC {test_metrics['roc_auc']:.3f}. Only {metrics['dataset']['positive_rate']:.1%} of the wines are good quality, so class weights were used during training.")

    st.markdown('**Classification report**')
    st.code(metrics['classification_report'], language='text')

    st.markdown('**Training curves**')
    plot_left, plot_right = st.columns(2)
    if os.path.exists(LOSS_PLOT_PATH):
        plot_left.image(LOSS_PLOT_PATH, caption='Training vs validation loss', use_container_width=True)
    if os.path.exists(ACCURACY_PLOT_PATH):
        plot_right.image(ACCURACY_PLOT_PATH, caption='Training vs validation accuracy', use_container_width=True)
    if os.path.exists(CONFUSION_MATRIX_PATH):
        st.image(CONFUSION_MATRIX_PATH, caption='Confusion matrix (test split)', width=420)
