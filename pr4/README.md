# CIFAR-10 Image Classification with CNNs

This project is a Jupyter notebook walkthrough of image classification with
TensorFlow/Keras. It uses the CIFAR-10 dataset to explore how convolutional
neural networks process images, then compares several model-building
approaches: a LeNet-5-style network, a custom CNN, data augmentation, and
MobileNetV2 transfer learning. The notebook ends with a small standalone
residual-block example.

## Project contents

| File | Description |
| --- | --- |
Video Link :- https://drive.google.com/file/d/1HYD7H8auptV9xWH9vlluY6ERNEEq_N7w/view?usp=sharing
| [`pr4.ipynb`](./pr4.ipynb) | The complete, executable notebook, including explanations, code, visualizations, training, and evaluation.|

All plots are displayed in the notebook. The notebook does not require
separate project data files; Keras downloads CIFAR-10 on first use. The
transfer-learning section also requests pretrained ImageNet weights for
MobileNetV2.

## Dataset

CIFAR-10 contains 60,000 color images, each 32 × 32 pixels, divided into 10
classes. Keras supplies 50,000 training images and 10,000 test images.

| Label | Class | Label | Class |
| ---: | --- | ---: | --- |
| 0 | Airplane | 5 | Dog |
| 1 | Automobile | 6 | Frog |
| 2 | Bird | 7 | Horse |
| 3 | Cat | 8 | Ship |
| 4 | Deer | 9 | Truck |

The notebook visualizes examples from each class, considers visually similar
classes, and inspects pixel-value distributions before and after
normalization.

## What the notebook covers

1. **Load and inspect data** — Load CIFAR-10 through `keras.datasets`, examine
   array shapes and data types, and visualize sample images.
2. **Prepare inputs and labels** — Convert image pixels to `float32` in the
   `[0, 1]` range and one-hot encode labels for 10-class softmax
   classification.
3. **Understand convolution** — Demonstrate a convolution filter and compare
   `VALID` versus `SAME` padding and different stride settings.
4. **Explore CNN feature extraction** — Visualize feature maps, compare
   max-pooling with average-pooling, and inspect flattening and output shapes.
5. **Build baseline architectures** — Implement a LeNet-5-style model and a
   custom CNN from scratch.
6. **Train and analyze models** — Train with categorical cross-entropy and
   Adam, use validation data, and examine learning curves, classification
   metrics, confusion matrices, and misclassified examples.
7. **Compare augmentation methods** — Inspect transformations from
   `ImageDataGenerator` and compare the baseline CNN with augmented models,
   including Keras preprocessing layers.
8. **Apply transfer learning** — Resize images to 96 × 96, use ImageNet-
   pretrained MobileNetV2 as a frozen feature extractor, then fine-tune its
   final 30 layers.
9. **Inspect a residual connection** — Construct a minimal Functional API
   residual block that adds a shortcut projection to a two-convolution main
   path. This is a demonstration and is not part of the notebook's model
   comparison.

The notebook's training cells use short demonstration runs (generally two
epochs for the scratch CNN and feature-extraction experiments, and one epoch
for MobileNetV2 fine-tuning). Consequently, results are useful for following
the workflow, not as benchmark-quality accuracy claims. Accuracy and other
metrics are calculated when the notebook runs; they can vary with
initialization, framework version, and hardware.

## Requirements

- Python version supported by the TensorFlow release installed in your
  environment
- Jupyter Notebook or JupyterLab
- TensorFlow (the notebook uses `tf.keras`)
- NumPy
- Matplotlib
- SciPy
- scikit-learn
- pandas
- seaborn

The notebook uses a CPU by default if no compatible GPU is available. A GPU is
optional, but can substantially reduce training time. Running every section,
especially the transfer-learning section, can require significant memory:
resizing the full CIFAR-10 training and test arrays to 96 × 96 creates large
in-memory tensors.

## Setup and run

Open PowerShell in this project directory and create an environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install tensorflow numpy matplotlib scipy scikit-learn pandas seaborn jupyter
```

Start Jupyter:

```powershell
jupyter notebook
```

Open `pr4.ipynb`, select the environment's Python kernel, then choose
**Kernel → Restart Kernel and Run All Cells** (or run the cells from top to
bottom). The first execution may take longer while Keras downloads CIFAR-10
and MobileNetV2 pretrained weights. Keep the notebook directory writable so
Jupyter can save the executed notebook and cell outputs.

If TensorFlow cannot be installed, verify that your Python version and
operating system are supported by the TensorFlow version you selected. For
GPU use, install and configure the appropriate TensorFlow/CUDA dependencies
for your platform; GPU setup is not required to run the project.

## Models and evaluation

### LeNet-5-style baseline

The LeNet-5-inspired Sequential model uses two convolution and average-pooling
stages, followed by a flatten layer and dense layers with 120 and 84 units.
Its 10-unit softmax layer produces class probabilities.

### Custom CNN

The scratch CNN uses a 32-filter convolution block, a 64-filter block, and a
128-filter convolution, followed by global average pooling and a dense
classification head. The notebook inspects its parameter count and trains it
with Adam and categorical cross-entropy. Validation loss is monitored with
early-stopping and learning-rate-reduction callbacks.

### Augmented CNNs

Two augmentation workflows are compared with the unaugmented CNN:

- `ImageDataGenerator` applies rotations, shifts, horizontal flips, and zoom.
- Keras preprocessing layers apply random flips, rotations, zoom, and
  translations within the model.

The models are evaluated on the original, unaugmented test set.

### MobileNetV2 transfer learning

Images are resized to 96 × 96 for MobileNetV2. The notebook first freezes the
ImageNet-pretrained convolutional base and trains a new classification head.
It then unfreezes the final 30 base-model layers and fine-tunes with a smaller
learning rate. The notebook plots the combined feature-extraction and
fine-tuning histories.

### Evaluation outputs

When run, the notebook produces model summaries, training/validation
visualizations, prediction and error-analysis outputs, and a comparison of
test accuracy across the architectures and augmentation/transfer-learning
variants. Numerical results are intentionally generated at runtime rather
than listed as fixed values in this README.

## Reproducibility and resource notes

- The notebook does not set random seeds, so training results may differ
  between runs.
- CIFAR-10 and ImageNet weights require an internet connection the first time
  they are downloaded; cached downloads may be reused afterward.
- The notebook keeps several image arrays and models in memory. If memory is
  limited, run only the sections you need and restart the kernel between
  experiments.
- The training epochs are deliberately short for demonstration. Increase the
  epoch counts and review the validation curves before drawing conclusions
  about comparative model performance.
