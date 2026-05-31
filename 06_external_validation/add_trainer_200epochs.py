# add_trainer_200epochs.py
# Adds nnUNetTrainer_200epochs class to the installed nnunetv2 package.
# Run once: python add_trainer_200epochs.py

import os

path = os.path.join(
    os.path.dirname(__import__("nnunetv2").__file__),
    "training", "nnUNetTrainer", "variants", "training_length",
    "nnUNetTrainer_Xepochs.py"
)

with open(path, "r") as f:
    content = f.read()

if "nnUNetTrainer_200epochs" in content:
    print("Already exists — nothing to do.")
else:
    addition = """

class nnUNetTrainer_200epochs(nnUNetTrainer):
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = 200
"""
    with open(path, "a") as f:
        f.write(addition)
    print(f"Added nnUNetTrainer_200epochs to:\n  {path}")

    # verify
    with open(path, "r") as f:
        check = f.read()
    print("Verified in file:", "nnUNetTrainer_200epochs" in check)

# clear pycache
import shutil
cache_dir = os.path.join(os.path.dirname(path), "__pycache__")
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)
    print("Cleared __pycache__")