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
│   ├── experiment/
│   │   ├── example_experiment.yaml
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

### **6️⃣ Outputs**

All run outputs are saved under the `outputs/` directory:

```plaintext
outputs/
├── train/    ← training logs, checkpoints, config snapshots
├── eval/     ← evaluation logs and results
├── predict/  ← prediction files (e.g. .csv, .json)
```

Each run is timestamped for easy tracking and reproducibility.

## Contributing & Feedback

⚠️ I welcome contributions! If you spot any issues, have suggestions, or want to enhance the project, feel free to check the issue list, fork the repo, create a PR, or open a new issue. Your feedback is valuable in making this project more useful for everyone!

## 📜 License

This project is licensed under the MIT License.
