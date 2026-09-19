"""Train a deep neural network (MLP) to classify red wine as good quality or not.

Run with:  python train_model.py

Artifacts written to artifacts/:
  wine_quality_dnn.keras  - trained Keras model
  scaler.pkl              - StandardScaler fitted on the training split only
  metrics.json            - architecture summary and the metrics measured on the test split
  accuracy_plot.png       - training vs validation accuracy
  loss_plot.png           - training vs validation loss
  confusion_matrix.png    - confusion matrix on the test split
"""

import json
import os

import joblib
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, 'artifacts')
DATASET_PATH = os.path.join(BASE_DIR, 'winequality-red.csv')

FEATURE_ORDER = [
    'fixed acidity',
    'volatile acidity',
    'citric acid',
    'residual sugar',
    'chlorides',
    'free sulfur dioxide',
    'total sulfur dioxide',
    'density',
    'pH',
    'sulphates',
    'alcohol',
]

RANDOM_SEED = 42
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
DROPOUT_RATE = 0.3


def load_data():
    """Load the dataset and binarise the target the same way the original project did."""
    df = pd.read_csv(DATASET_PATH)
    X = df[FEATURE_ORDER].astype('float32')
    # good quality wine == quality >= 7, matching the original notebook/app target
    y = (df['quality'] >= 7).astype('int32')
    return X, y


def split_and_scale(X, y):
    """Split into train/validation/test, then scale.

    The scaler is fitted on the training split only and merely applied to the
    validation and test splits, so no statistics from held-out data leak into training.
    """
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.20, random_state=RANDOM_SEED, stratify=y_train_full
    )

    scaler = StandardScaler().fit(X_train)
    return (
        scaler.transform(X_train),
        scaler.transform(X_val),
        scaler.transform(X_test),
        y_train.to_numpy(),
        y_val.to_numpy(),
        y_test.to_numpy(),
        scaler,
    )


def build_model(n_features):
    """Dense feed-forward network for tabular data.

    ReLU in the hidden layers keeps gradients from saturating (unlike sigmoid/tanh,
    whose derivatives vanish for large |x|) and makes the network cheap to train.
    The single sigmoid output squashes the logit into [0, 1] so it can be read as
    P(good quality), which is exactly what binary cross-entropy expects.
    BatchNormalization stabilises the activation distribution between layers and
    Dropout randomly zeroes units during training to limit overfitting on this
    small (1599-row) dataset.
    """
    model = keras.Sequential(
        [
            keras.Input(shape=(n_features,), name='wine_features'),
            layers.Dense(128, activation='relu', name='hidden_1'),
            layers.BatchNormalization(name='batch_norm'),
            layers.Dropout(DROPOUT_RATE, name='dropout'),
            layers.Dense(64, activation='relu', name='hidden_2'),
            layers.Dense(32, activation='relu', name='hidden_3'),
            layers.Dense(1, activation='sigmoid', name='good_quality_probability'),
        ],
        name='wine_quality_dnn',
    )

    # Binary cross-entropy is the negative log-likelihood of a Bernoulli target:
    # -[y*log(p) + (1-y)*log(1-p)]. It penalises confident wrong predictions heavily
    # and pairs with the sigmoid output to give a well-behaved gradient (p - y).
    #
    # Adam keeps per-parameter running averages of the gradient (first moment) and of
    # the squared gradient (second moment) and updates weights with
    #   w <- w - lr * m_hat / (sqrt(v_hat) + eps)
    # so each weight gets its own effective step size; that adaptivity plus momentum
    # converges much faster than plain SGD on this small tabular problem.
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )
    return model


def plot_history(history):
    for metric, filename, title in [
        ('accuracy', 'accuracy_plot.png', 'Training vs validation accuracy'),
        ('loss', 'loss_plot.png', 'Training vs validation loss'),
    ]:
        plt.figure(figsize=(7, 4.5))
        plt.plot(history.history[metric], label=f'train {metric}')
        plt.plot(history.history[f'val_{metric}'], label=f'validation {metric}')
        plt.xlabel('epoch')
        plt.ylabel(metric)
        plt.title(title)
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(ARTIFACTS_DIR, filename), dpi=120)
        plt.close()


def plot_confusion_matrix(cm):
    plt.figure(figsize=(5, 4.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Reds',
        xticklabels=['not good', 'good'],
        yticklabels=['not good', 'good'],
    )
    plt.xlabel('predicted')
    plt.ylabel('actual')
    plt.title('Confusion matrix (test split)')
    plt.tight_layout()
    plt.savefig(os.path.join(ARTIFACTS_DIR, 'confusion_matrix.png'), dpi=120)
    plt.close()


def main():
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    keras.utils.set_random_seed(RANDOM_SEED)

    X, y = load_data()
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = split_and_scale(X, y)
    print(f'train={len(X_train)} validation={len(X_val)} test={len(X_test)}')

    model = build_model(X_train.shape[1])
    model.summary()

    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=15, restore_best_weights=True
    )

    # model.fit runs the full training loop: each batch is pushed through the network
    # (forward propagation) to produce predictions and a loss, TensorFlow records those
    # ops on a GradientTape, then backpropagation walks the tape in reverse applying the
    # chain rule to get dLoss/dWeight for every parameter, and Adam applies those
    # gradients to update the weights. Repeat for every batch of every epoch.
    #
    # Only ~13.6% of the wines are "good", so class weights scale the loss of the
    # minority class up; without them the network can score well by predicting
    # "not good" for everything.
    class_counts = np.bincount(y_train)
    class_weight = {
        0: len(y_train) / (2.0 * class_counts[0]),
        1: len(y_train) / (2.0 * class_counts[1]),
    }

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping],
        class_weight=class_weight,
        verbose=2,
    )

    probabilities = model.predict(X_test, verbose=0).ravel()
    y_pred = (probabilities >= 0.5).astype('int32')

    report = classification_report(y_test, y_pred, target_names=['not good', 'good'])
    cm = confusion_matrix(y_test, y_pred)
    print(report)
    print(cm)

    plot_history(history)
    plot_confusion_matrix(cm)

    epochs_completed = len(history.history['loss'])
    metrics = {
        'dataset': {
            'rows': int(len(X)),
            'train': int(len(X_train)),
            'validation': int(len(X_val)),
            'test': int(len(X_test)),
            'positive_rate': float(y.mean()),
        },
        'architecture': [
            'Input: 11 chemical features',
            'Dense 128, ReLU',
            'BatchNormalization',
            f'Dropout {DROPOUT_RATE}',
            'Dense 64, ReLU',
            'Dense 32, ReLU',
            'Dense 1, Sigmoid',
        ],
        'hidden_layers': 3,
        'total_parameters': int(model.count_params()),
        'loss_function': 'binary_crossentropy',
        'optimizer': 'Adam',
        'learning_rate': LEARNING_RATE,
        'batch_size': BATCH_SIZE,
        'max_epochs': EPOCHS,
        'epochs_completed': epochs_completed,
        'early_stopping': 'monitor=val_loss, patience=15, restore_best_weights=True',
        'test_metrics': {
            'accuracy': float(accuracy_score(y_test, y_pred)),
            'precision': float(precision_score(y_test, y_pred, zero_division=0)),
            'recall': float(recall_score(y_test, y_pred, zero_division=0)),
            'f1': float(f1_score(y_test, y_pred, zero_division=0)),
            'roc_auc': float(roc_auc_score(y_test, probabilities)),
        },
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'feature_order': FEATURE_ORDER,
    }

    model.save(os.path.join(ARTIFACTS_DIR, 'wine_quality_dnn.keras'))
    joblib.dump(scaler, os.path.join(ARTIFACTS_DIR, 'scaler.pkl'))
    with open(os.path.join(ARTIFACTS_DIR, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics['test_metrics'], indent=2))
    print(f'artifacts written to {ARTIFACTS_DIR}')


if __name__ == '__main__':
    main()
