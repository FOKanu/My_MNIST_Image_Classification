# MNIST Image Classification — CNN in TensorFlow/Keras

A small, reproducible handwritten-digit classifier: a custom CNN trained on MNIST, with a scripted training/evaluation pipeline, a saved model, and an analysis notebook.

![Python](https://img.shields.io/badge/Python-3.12-blue) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.16-orange) ![License](https://img.shields.io/badge/License-GPL--3.0-green)

## Results

From the committed model metadata (`models/model_metadata.json`, saved alongside `models/mnist_cnn_final.h5`):

| Metric | Value |
|---|---|
| Test accuracy | **97.89%** |
| Test loss | 0.0695 |
| Parameters | 7,184 |
| Training config | 5 epochs, batch 32, Adam, 30% validation split, early stopping + checkpointing |

The notebook's training run (`notebooks/mnist_classification.ipynb`) reaches ~97.7% training accuracy, consistent with the saved model's test result.

## Architecture

```text
Input 28×28×1
  → Conv2D(8, 4×4, same) → MaxPool2D
  → Conv2D(16, 3×3)      → MaxPool2D
  → Flatten → Dense(10, ReLU) → Dense(10, softmax)
```

Defined in `src/model.py` (`MNISTCNN`), with preprocessing in `src/data_preprocessing.py` (normalisation to [0,1], channel dimension, one-hot labels) and evaluation in `src/evaluation.py` (confusion matrix, per-class analysis).

## Run it

```bash
pip install -r requirements.txt
python main.py        # loads MNIST, trains, evaluates, saves models/mnist_cnn_final.h5
python test_setup.py  # quick import/environment sanity check
```

Or explore `notebooks/mnist_classification.ipynb` for the step-by-step version with plots (training curves, confusion matrix, sample predictions).

## Scope — what this is and isn't

- A learning project demonstrating a clean CNN implementation, a modular `src/` layout (preprocessing / model / evaluation), and disciplined model bookkeeping (architecture + metrics recorded in `models/model_metadata.json`).
- It is **not** deployed: there is no API or web app in this repo. (`flask`/`gunicorn` in `requirements.txt` are marked optional, for a possible future serving layer.)
- MNIST is a solved benchmark — 97.89% is a solid result for a 7k-parameter model, not a research claim.

## Author

Francis Okorie Kanu

## License

GPL-3.0 — see [LICENSE](LICENSE).
