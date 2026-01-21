#!/bin/bash

# ===========================================
# builder.sh
# Create project folder structure (no .pyc)
# ===========================================

set -e

echo "[builder] Creating directory structure..."

# Root-level files
touch main.py
touch builder.sh
touch run_train.sh

# Directories
mkdir -p checkpoints
mkdir -p logs

# -------------------------
# config/
# -------------------------
mkdir -p config
touch config/__init__.py
touch config/model_config.py
touch config/train_config.py
touch config/data_config.py

# -------------------------
# data/
# -------------------------
mkdir -p data
touch data/__init__.py
touch data/dataset.py
touch data/collator.py
touch data/preprocess.py

# -------------------------
# models/
# -------------------------
mkdir -p models
touch models/__init__.py
touch models/abstract.py
touch models/base.py
touch models/utils.py
touch models/_layers.py
touch models/transformer.py

# -------------------------
# trainers/
# -------------------------
mkdir -p trainers
touch trainers/__init__.py
touch trainers/abstract.py
touch trainers/trainer.py

# trainers/callbacks
mkdir -p trainers/callbacks
touch trainers/callbacks/__init__.py
touch trainers/callbacks/base.py
touch trainers/callbacks/checkpoint.py
touch trainers/callbacks/logging.py
touch trainers/callbacks/early_stopping.py

# -------------------------
# utils/
# -------------------------
mkdir -p utils
touch utils/__init__.py
touch utils/distributed.py
touch utils/logging.py
touch utils/checkpoint.py

echo "[builder] Project structure created successfully!"
