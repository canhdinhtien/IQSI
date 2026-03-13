# How to use:

## Installation

```bash
git clone https://github.com/canhdinhtien/IQSI
cd IQSI
pip install -e .
export WANDB_API_KEY=your_api_key
find synthetic_data/ -name ".ipynb_checkpoints" -type d -exec rm -rf {} +
```

## USAGE

```bash
accelerate launch main.py
or
python3 -m accelerate.commands.launch main.py

```

> [!IMPORTANT]  
> After the `real_data` directory is created, you **must move** `split_coop.csv` to `real_data/train/` before training.
