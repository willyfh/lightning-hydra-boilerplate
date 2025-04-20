<div align="center">
<h1> ⚡ Lightning-Hydra-Boilerplate </h1>

[![python](https://img.shields.io/badge/python-3.10-blue)]() [![python](https://img.shields.io/badge/python-3.11-blue)]() [![python](https://img.shields.io/badge/python-3.12-blue)]() [![Run Tests](https://github.com/willyfh/lightning-hydra-boilerplate/actions/workflows/ci-checks.yaml/badge.svg)](https://github.com/willyfh/lightning-hydra-boilerplate/actions/workflows/ci-checks.yaml) [![codecov](https://codecov.io/gh/willyfh/lightning-hydra-boilerplate/graph/badge.svg?token=OGLCMT2KQ4)](https://codecov.io/gh/willyfh/lightning-hydra-boilerplate) [![MIT License](https://img.shields.io/badge/License-MIT-yellow)](https://opensource.org/licenses/MIT)

</div>

A project boilerplate for deep learning experiments using PyTorch Lightning and Hydra, designed for rapid prototyping, clean configuration management, and scalable research workflows.

🔬 This project will continue to be validated through research workflows. While the current functionality is ready for use, ongoing improvements are expected as it evolves through practical use and feedback 🌱.

🚀 **Feel free to click "Use this template"** to start your own project based on this boilerplate!

## 🔑 Key Features

| Feature                      | Description                                                                            |
| ---------------------------- | -------------------------------------------------------------------------------------- |
| 📝 **Configurable Setup**    | Model, dataset, and training configuration using `Hydra`.                              |
| 📊 **Logging**               | Default: `TensorBoard`. Easily switch to other tools via `PyTorch Lightning`.          |
| ⚗️ **Hyperparameter Tuning** | Integration with `Optuna` for automated hyperparameter search.                         |
| 🧑‍💻 **Callbacks**             | Includes early stopping, checkpointing (best & last), and more via `PyTorch Lightning` |
| 💡 **Accelerator Support**   | Example configs for `CPU`, `DDP`, `DeepSpeed`, and more via `PyTorch Lightning`        |
| 🎯 **Scripts**               | Pre-configured training, evaluation, and prediction workflows.                         |
| 📂 **Organized Outputs**     | Logs, checkpoints, configs, and predictions saved in a structured folder.              |
| 🔍 **Metrics Handling**      | Clean separation of metrics, works with `TorchMetrics` or custom.                      |
| 🔧 **Dependency Management** | Uses `Poetry` for environment and package management.                                  |
| ⚙️ **CI Integration**        | Comes with pre-commit hooks and automated testing setup.                               |

## 📁 Project Structure

```plaintext
lightning-hydra-boilerplate/
│── configs/                  # YAML configurations for Hydra
│   ├── data/
│   │   ├── example_data.yaml
│   ├── model/
│   │   ├── example_model.yaml
│   ├── trainer/
│   │   ├── base.yaml
│   │   ├── cpu.yaml
│   │   ├── ddp.yaml
│   │   ├── deepspeed.yaml
│   │   ├── ...
│   ├── experiment/
│   │   ├── example_experiment.yaml
│   ├── params_search/
│   │   ├── example_optuna.yaml
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
│   ├── callbacks/
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
 poetry run python src/train.py experiment=example_experiment
```

This will:

- Train the model on the training set
- Validate during training using the validation set
- Test the final model on the test set (unless `skip_test=true` is set)

### **3️⃣ Evaluate a Model**

If you need to run evaluation independently on a specific checkpoint and dataset split (val or test), use the evaluation script:

```bash
poetry run python src/eval.py experiment=example_experiment data_split=test ckpt_path=/path/to/checkpoint.ckpt
```

### **4️⃣ Predict Outputs**

To generate predictions from a trained model (e.g., for submission or analysis), run the predict.py script with the path to your checkpoint:

```bash
poetry run python src/predict.py experiment=example_experiment ckpt_path=/path/to/checkpoint.ckpt
```

This will:

- Run inference on the prediction set (defaults to `test` set)
- Save the results to a file (e.g., predictions.csv or predictions.json depending on `save_format`) under the Hydra run directory

### **5️⃣ Experiment Configuration**

All experiment settings are managed with Hydra.

You can modify the configs under configs/ or override them via CLI. For example:

```bash
 poetry run python src/train.py experiment=example_experiment trainer.max_epochs=10
```

More info: https://hydra.cc/docs/intro/

### 6️⃣ **Hyperparameter Optimization with Optuna**

To perform hyperparameter optimization using Optuna, you can run the training with the sweep feature, which allows you to search for the best hyperparameters for your model automatically. The results will be stored in a designated sweeps/ directory.

```bash
poetry run python src/train.py experiment=example_experiment params_search=example_optuna
```

This will:

- Perform hyperparameter optimization using Optuna

- Log the best trial configuration and metrics to `outputs/sweeps/`

### **7️⃣ Outputs**

All run outputs are saved under the `outputs/` directory:

```plaintext
outputs/
├── train/    ← training logs, checkpoints, config snapshots
├── eval/     ← evaluation logs and results
├── predict/  ← prediction files (e.g. .csv, .json)
├── sweeps/   ← results from Optuna hyperparameter sweeps
```

Each run is timestamped for easy tracking and reproducibility. The `sweeps/` directory will contain subdirectories for each trial and the results of the hyperparameter search.

## 🌱 Contributing & Feedback

If you spot any issues, have suggestions, or want to enhance the project, feel free to check the issue list, fork the repo, create a PR, or open a new issue. Your feedback is valuable in making this project more useful for everyone!

## 📜 License

This project is licensed under the MIT License.
