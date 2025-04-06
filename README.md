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
│   ├── config.yaml
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

You can override configs using Hydra, for example:

```bash
 poetry run python src/train.py trainer.max_epochs=10
```

### **3️⃣ Experiment Configuration**

All experiment settings are managed with Hydra.
Modify `configs/config.yaml` or override via CLI. See for more details: https://hydra.cc/docs/intro/

### **4️⃣ Outputs**

#### Training outputs

- **Training logs** (using **TensorBoard** by default) can be found in:
  `train_output/{experiment_name}-{timestamp}/logs/`.

- **Hydra** stores the training config snapshots, in:
  `train_output/{experiment_name}-{timestamp}/.hydra/`.

- **Checkpoints** (including both best and last models) are saved in:
  `train_output/{experiment_name}-{timestamp}/checkpoints/`.

#### Evaluation outputs

- **Evaluation logs** (using **TensorBoard** by default) can be found in:
  `eval_output/{experiment_name}-{timestamp}/logs/`.

- **Hydra** stores evaluation config snapshots, in:
  `eval_output/{experiment_name}-{timestamp}/.hydra/`.

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

## 📝 TODO List

⚠️ _Feel free to fork the repo, create a PR, or open an issue if you spot anything or have ideas. I’d love to hear your feedback and make this more useful for everyone!_

- [ ] 🏆 Evaluation script
- [ ] Inference script
- [ ] 🛠 Hyperparameter tuning with Optuna
- [ ] 🚀 Check Multi-GPU
- [ ] ⚡ Add more Lightning Trainer features (resume, callbacks, etc.)
- [ ] 📈 MLflow and/or Wandb
- [ ] 🐳 Docker support for easy deployment
- [ ] Make metrics configurable

## 📜 License

This project is licensed under the MIT License.
