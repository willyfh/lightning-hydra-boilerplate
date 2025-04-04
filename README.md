# Lightning-Hydra-Boilerplate

A boilerplate training framework for deep learning experiments with **PyTorch Lightning** and **Hydra**.

## 📁 Project Structure

```plaintext
lightning-hydra-boilerplate/
│── configs/                  # YAML configurations for Hydra
│   ├── data/
│   │   ├── example_data.yaml  # Example dataset configuration
│   ├── logger/
│   │   ├── tensorboard.yaml   # Logger configuration
│   ├── model/
│   │   ├── example_model.yaml # Example model configuration
│   ├── trainer/
│   │   ├── default.yaml       # Default training config
│   ├── config.yaml            # Main config file
│
│── scripts/                   
│   ├── train.py               # Training script
│
│── src/                       # Source code
│   ├── data/
│   │   ├── example_data/
│   │   │   ├── lightning_datamodule.py  # Lightning DataModule
│   │   │   ├── torch_dataset.py         # Custom PyTorch Dataset
│   ├── model/
│   │   ├── example_model/
│   │   │   ├── lightning_module.py  # PyTorch Lightning Model
│   │   │   ├── torch_model.py       # Custom PyTorch Model
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
 poetry run python scripts/train.py
```

You can override configs using Hydra, for example:

```bash
 poetry run python scripts/train.py trainer.max_epochs=10
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

## 📝 TODO List
- [ ] 🔄 Configurable Callbacks (EarlyStopping, ModelCheckpoint, etc.)
- [ ] 🛠 Hyperparameter tuning with Optuna
- [ ] 🏆 Evaluation script
- [ ] 🚀 Check Multi-GPU
- [ ] 📈 MLflow and/or Wandb
- [ ] ✅ Unit tests
- [ ] 🐳 Docker support for easy deployment
- [ ] ⚙️ Continuous Integration (CI) setup
- [ ] 📂 Organize `logs/` (Lightning) and `outputs/` (Hydra) properly

## 📜 License
This project is licensed under the MIT License.
