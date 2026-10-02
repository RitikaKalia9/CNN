# CNN  — Convolutional Neural Networks with PyTorch

**Notebook:** `CNN.ipynb`
**Companion manual:** *Convolutional Neural Networks: Hands-on with PyTorch* (lab manual)
**Framework:** PyTorch · **Dataset:** CIFAR-10

## Overview

A step-by-step practical that takes the CNN lecture material into code. Each quantity is first worked out by hand, then reproduced with PyTorch, and the two are checked against each other. The notebook ends with a small CIFAR-10 classifier, ten lab exercises and eight viva questions.

## What it covers

| Steps | Topic |
|---|---|
| 1–2 | Setup, device selection, the N, C, H, W tensor layout |
| 3 | Convolution (cross-correlation) by hand vs `F.conv2d` |
| 4 | Stride, padding and the output-size formula `⌊(H + 2P − K)/S⌋ + 1` |
| 5 | ReLU and its gradient |
| 6–7 | Max vs average pooling, and how gradients flow back through each |
| 8 | Backpropagation through a convolution (`dL/dK`, `dL/dX`, `dL/db`) |
| 9 | Parameter count, receptive field, MACs |
| 10 | Load CIFAR-10 and display a batch |
| 11–12 | Build `SmallCNN` and trace a tensor through it |
| 13–15 | Train (5 epochs, Adam, lr 1e-3), evaluate, per-class accuracy |
| 16–17 | Visualize learned conv1 filters and feature maps |

### Model: `SmallCNN`

```
Input 3×32×32
→ Conv(3→16, 3×3, pad 1) → ReLU → MaxPool 2×2   (16×16×16)
→ Conv(16→32, 3×3, pad 1) → ReLU → MaxPool 2×2  (32×8×8)
→ Flatten (2048) → Linear(2048→128) → ReLU → Linear(128→10)
```

Logits are returned without softmax, because `CrossEntropyLoss` applies it internally.

## Lab exercises (in the notebook)

1. Change the Step 3 kernel to all ones
2. Output-size checks for `(32,4,1,1)` and `(7,3,2,0)`
3. Pooling gradients with an all-ones upstream gradient
4. A 3×3 kernel on a 4×4 input
5. Add a third `32→64` conv and pooling layer
6. Replace max pooling with average pooling
7. Set `padding=0` and trace shapes
8. Train for 15 epochs and plot loss and accuracy
9. Add `RandomHorizontalFlip` to the training transform only
10. Compare MACs with parameter counts for `conv1` and `conv2`

The notebook also has a quick revision table, a conclusion and the viva questions.

## Results recorded in the notebook

| Item | Result |
|---|---|
| Hand convolution vs `F.conv2d` | identical (`[[2, 4], [-1, 0]]`) |
| Output-size formula | matches PyTorch in all four test cases |
| Parameters, 3×3 conv 32→64 | 18,496 |
| Receptive field | 5 (two 3×3 layers), 7 (three 3×3 layers) |
| `SmallCNN` test accuracy (5 epochs) | 66.16% |
| Average-pool variant (5 epochs) | about 64% test accuracy after epoch 4 |
| Flip augmentation (5 epochs) | train accuracy 70.98% at epoch 5 |
| `conv1` / `conv2` MACs | 442,368 / 1,179,648 |
| `conv1` / `conv2` parameters | 448 / 4,640 |

## Known issue to check

The Step 15 per-class accuracies (for example `car 0.0%`, `truck 0.0%`) do not agree with the overall 66.16% from Step 14. If the classes were evaluated correctly they would average close to 66%. Re-run Steps 11–15 from top to bottom with a fresh kernel. If the numbers still disagree, check that `model` has not been reassigned between the two cells.

## Requirements

- Python 3.9+
- `torch`, `torchvision`, `matplotlib`
- CPU is enough; a GPU only shortens training
- CIFAR-10 (about 170 MB) downloads automatically to `./data`

```bash
pip install torch torchvision matplotlib
```

## How to run

1. Open `CNN.ipynb` in Jupyter, JupyterLab or Google Colab.
2. Run the cells in order. Later steps reuse the setup, data and model cells.
3. Write down your prediction before running any cell that asks for one, then compare.
