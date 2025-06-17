# 🎨 ReFaceIt: Sketch-to-Face Image Translation using DCGAN

This project implements a Deep Convolutional GAN (DCGAN) for translating **sketch images** into realistic **face photographs**, trained on the CUHK dataset. Two versions of the model are provided — one trained with **Binary Cross Entropy (BCE)** loss and another with **Mean Squared Error (MSE)** loss — to analyze and compare reconstruction quality.

---

## 📁 Directory Structure

```

ReFaceIt/
│
├── CUHK/                          # Dataset folder
│   ├── checkpoint\_bce/               # Checkpoints for BCE model training
|   ├── results/                      # Results of MSE-based DCGAN
|   ├── results\_bce/                  # Results of BCE-based DCGAN
    ├── Training\_photo/               # Real face images for training
    ├── Training\_sketch/             # Sketches for training
    ├── Testing\_photo/               # Real face images for testing
    └── Testing\_sketch/              # Sketches for testing 
│
├── DCGAN\_BCE.py              # DCGAN with BCE loss
│   ├── DCGAN.py                  # DCGAN with MSE loss
│   ├── final\_losses\_BCE.png      # Loss curve for BCE model
│   ├── final\_losses\_MSE.png      # Loss curve for MSE model
│   ├── generator\_epoch\_100.pth   # Saved generator model
│   └── README.md                 # Project README

````

---

## 🔍 Project Overview

- **Objective:** Generate realistic face photos from input sketches using GANs.
- **Dataset:** CUHK Face Sketch dataset
- **Models Used:** DCGAN architecture
- **Loss Functions Compared:**
  - Binary Cross Entropy (BCE)
  - Mean Squared Error (MSE)

---

## 🛠️ Requirements

Install the required libraries with:

```bash
pip install torch torchvision numpy matplotlib scikit-image opencv-python
````

---

## 🚀 How to Run

### 1. Train with BCE Loss

```bash
python CUHK/DCGAN_BCE.py
```

### 2. Train with MSE Loss

```bash
python CUHK/DCGAN.py
```

---

## 📊 Results & Evaluation

* **Loss Curves:**

  * `final_losses_BCE.png` – Training loss for BCE model
  * `final_losses_MSE.png` – Training loss for MSE model

* **Quantitative Metric:**
  Structural Similarity Index (SSIM)

  * **BCE Model SSIM:** 0.51
  * **MSE Model SSIM:** 0.55

* **Inference Outputs:**

  * Check `results/` and `results_bce/` folders for generated images.

---

## 📦 Pretrained Weights

* `generator_epoch_100.pth` – Saved generator weights after 100 epochs (likely MSE-based)

To load this:

```python
model.load_state_dict(torch.load('CUHK/generator_epoch_100.pth'))
```

---

## 🧠 Future Scope

1. **Multi-sample Generation**
   Enable the model to generate multiple photo reconstructions per sketch, letting users select the best one.

2. **Side-profile Reconstruction**
   Extend the model to handle side-profile sketches, enhancing practical use in law enforcement.

3. **GAN Enhancements**
   Explore improved architectures like U-GAT-IT or StyleGAN2 for better realism and control.

---

## 🤝 Acknowledgements

* CUHK Face Sketch dataset \[CUFS]
* DCGAN implementation inspired by PyTorch official tutorials

---

## 📜 License

This project is licensed under the MIT License.

```

---

