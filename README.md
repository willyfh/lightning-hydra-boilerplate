<div align="center">
<h1> ⚡ Lightning-Hydra-Boilerplate </h1>
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
│   │   ├── hydra_utils.py
│   ├── train.py               # Training entrypoint
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

### **4️⃣ Logging**

- Training logs are saved using **TensorBoard** by default. Logs can be found in `logs/`.
- Hydra stores experiment outputs, including config snapshots, in the `outputs/` directory.

## ✅ Completed Tasks

- [x] Initial project setup with PyTorch Lightning and Hydra
- [x] Basic training workflow with example model and data
- [x] TensorBoard logging support
- [x] Poetry setup
- [x] Configurable Callbacks (EarlyStopping, ModelCheckpoint, etc.)
- [x] Run test after training
- [x] Pre-commit setup

## 📝 TODO List

⚠️ _Feel free to fork the repo, create a PR, or open an issue if you spot anything or have ideas. I’d love to hear your feedback and make this more useful for everyone!_

- [ ] 🏆 Evaluation script
- [ ] 🛠 Hyperparameter tuning with Optuna
- [ ] 🚀 Check Multi-GPU
- [ ] 📈 MLflow and/or Wandb
- [ ] ✅ Unit tests + CI
- [ ] 🐳 Docker support for easy deployment
- [ ] 📂 Organize `logs/`, `checkpoints/` (Lightning), and `outputs/` (Hydra) properly
- [ ] 📝 Add logger

## 📜 License

This project is licensed under the MIT License.
