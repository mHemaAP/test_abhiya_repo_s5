Dog Breed Classification using PyTorch Lightning
Project Overview
This project implements a dog breed classification model using PyTorch Lightning, trained on the Dogs Breed Classification dataset from Kaggle. It features real-time progress visualization using Rich, model performance tracking with TensorBoard, and a Docker container for easy deployment.

Key Features
- PyTorch Lightning Framework: Utilizes PyTorch Lightning for efficient model training and management.
- Rich Progress Bar: Provides interactive and informative progress bars during training using Rich library.
- TensorBoard Integration: Enables real-time visualization of model metrics and performance through TensorBoard.
- Docker Containerization: Includes Docker setup for easy deployment and reproducibility across different environments.

Dataset
This project uses the Dogs Breed Classification dataset available on Kaggle. 
The dataset contains images of various dog breeds, which are used to train a classification model.


├── checkpoints/
├── data/
│   └── dogs_breed_classification/
├── data_module/
│   ├── dogs_datamodule.py
│   └── __init__.py
├── docker-compose.yml
├── Dockerfile
├── logs/
├── model/
│   ├── dogs_classifier.py
│   └── __init__.py
├── requirements.txt
├── utils/
│   ├── eval.py
│   ├── infer.py
│   └── train.py
└── README.md
