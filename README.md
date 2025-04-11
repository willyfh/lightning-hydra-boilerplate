<div align="center">
<h1> ⚡ Lightning-Hydra-Boilerplate </h1>

[![python](https://img.shields.io/badge/python-3.12-blue)]() [![Run Tests](https://github.com/willyfh/lightning-hydra-boilerplate/actions/workflows/pytest.yaml/badge.svg)](https://github.com/willyfh/lightning-hydra-boilerplate/actions/workflows/pytest.yaml)

</div>

A project boilerplate for deep learning experiments using PyTorch Lightning and Hydra, designed for rapid prototyping, clean configuration management, and scalable research workflows.

⚠️ _This project is in its early stages and still under active development. Expect breaking changes and incomplete features._

## 📁 Project Structure

```plaintext
lightning-hydra-boilerplate/
│── configs/                  # YAML configurations for Hydra
│   ├── data/
│   │   ├── example_data.yaml
│   ├── model/
│   │   ├── example_model.yaml
│   ├── trainer/
│   │   ├── default.yaml
│   ├── train.yaml
│   ├── eval.yaml
│   ├── predict.yaml
│
│── src/                       # Core codebase
│   ├── data/
│   │   ├── example_data/
│   │   │   ├── lightning_datamodule.py
│   │   │   ├── torch_dataset.py
│   ├── model/
│   │   ├── example_model/
│   │   │   ├── lightning_module.py
│   │   │   ├── torch_model.py
│   ├── utils/
│   ├── train.py               # Training entrypoint
│   ├── eval.py                # Evaluation entrypoint
│   ├── predict.py             # Inference entrypoint (for making predictions)
│
│── tests/                     # Unit tests
│
│── .gitignore
│── LICENSE
│── poetry.lock
│── pyproject.toml
│── README.md
```

## 🚀 Getting Started

### **1️⃣ Install Dependencies**

This project uses **Poetry** for dependency management.

```bash
pip install poetry  # If you haven't installed Poetry yet
poetry install
```

### **2️⃣ Train a Model**

Run the training script with the default configurations:

```bash
 poetry run python src/train.py
```

This will:

- Train the model on the training set
- Validate during training using the validation set
- Test the final model on the test set (unless `skip_test=true` is set)

### **3️⃣ Evaluate a Model**

If you need to run evaluation independently on a specific checkpoint and dataset split (val or test), use the evaluation script:

```bash
poetry run python src/eval.py data_split=test ckpt_path=/path/to/checkpoint.ckpt
```

### **4️⃣ Predict Outputs**

To generate predictions from a trained model (e.g., for submission or analysis), run the predict.py script with the path to your checkpoint:

```bash
poetry run python src/predict.py ckpt_path=/path/to/checkpoint.ckpt
```

This will:

- Run inference on the prediction set (defaults to `test` set)
- Save the results to a file (e.g., predictions.csv or predictions.json depending on `save_format`) under the Hydra run directory

### **5️⃣ Experiment Configuration**

All experiment settings are managed with Hydra.

You can modify the configs under configs/ or override them via CLI. For example:

```bash
 poetry run python src/train.py trainer.max_epochs=10
```

More info: https://hydra.cc/docs/intro/

### **6️⃣ Outputs**

All run outputs are saved under the `outputs/` directory:

```plaintext
outputs/
├── train/    ← training logs, checkpoints, config snapshots
├── eval/     ← evaluation logs and results
├── predict/  ← prediction files (e.g. .csv, .json)
```

Each run is timestamped for easy tracking and reproducibility.

## ✅ Completed Tasks

- [x] Initial project setup with PyTorch Lightning and Hydra
- [x] Basic training workflow with example model and data
- [x] TensorBoard logging support
- [x] Poetry setup
- [x] Configurable Callbacks (EarlyStopping, ModelCheckpoint, etc.)
- [x] Run test after training
- [x] Setup pre-commit setup
- [x] Setup tests
- [x] Setup dependabot
- [x] Add logger
- [x] Organize `logs/`, `checkpoints/` (Lightning), and `outputs/` (Hydra) properly
- [x] Evaluation script (`eval.py`)
- [x] Inference script (`predict.py`)
- [x] Make optimizer configurable

## 📝 TODO List

⚠️ _Feel free to fork the repo, create a PR, or open an issue if you spot anything or have ideas. I’d love to hear your feedback and make this more useful for everyone!_

- [ ] Hyperparameter tuning with Optuna
- [ ] Check Multi-GPU
- [ ] Add more Lightning Trainer features (resume, callbacks, etc.)
- [ ] MLflow and/or Wandb
- [ ] Docker support for easy deployment
- [ ] Make metrics configurable
- [ ] Add task-specific examples and configs (e.g., object detection, text classification, etc.)
- [ ] Add experiment configs for reusable training/eval setups (e.g., `configs/experiments/exp1.yaml`)

## 📜 License

This project is licensed under the MIT License.
