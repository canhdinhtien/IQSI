from setuptools import setup, find_packages

setup(
    name="iqsi",
    version="0.0.1",
    packages=["iqsi"],
    python_requires=">=3.12",
    install_requires=[
        "ml_collections",
        "accelerate",
        "bitsandbytes",
        "diffusers",
        "faiss-cpu",
        "wandb",
        "fire==0.7.0",
        "huggingface_hub",
        "joblib",
        "loralib",
        "munch",
        "numpy",
        "packaging",
        "peft",
        "Pillow",
        "PyYAML",
        "safetensors",
        "timm",
        "torch",
        "torchvision",
        "tqdm",
        "transformers",
        "flask",
        "clip @ git+https://github.com/openai/CLIP.git",
        "xformers"
    ]
)