# Unsupervised-Concept-Based-Explanations-For-Sentiment-Analysis
Learning Unsupervised Concept-Based Explanations for Sentiment Analysis. CS 224U final project.

## 1. Setup

### 1.1 Install Dependencies
- It's recommended you use [Anaconda](https://www.anaconda.com/products/distribution) for many dependencies. If you don't want to use Anaconda, see the commented out requirements in [requirements.txt](./requirements.txt).
- Run `pip install -r requirements.txt` to install extra dependencies.
- Also [install PyTorch](https://pytorch.org/get-started/locally/).

### 1.2 Download Dynasent
Download `dynasent_model1.bin` from https://drive.google.com/drive/folders/1dpKrjNJfAILUQcJPAFc5YOXUT51VEjKQ and put it in the `models` folder in this directory.

## 2. Testing

### 2.1 Train and Save Models
Run [train_and_save_models.ipynb](./train_and_save_models.ipynb) to train 4 models (RoBERTa-Base, DynaSent Model 1, and a version of each made into an Attention-Abstraction Network) on the training data (DynaSent Round 2). This will fine-tune these models for sentiment analysis and save the resulting experiment dicts (which contain the best trained model in the `'model'` value) to a local `models` folder.

### 2.2 Evaluate Trained Models
Run [evaluate_trained_models.ipynb](./evaluate_trained_models.ipynb) to evaluate the trained models. This will plot their training and compare their performance on the held-out DynaSent test set as well as demonstrating some concept-based explanations for certain test examples.

## 3. Attribution

- https://arxiv.org/abs/2004.13003 for proposing the abstraction-aggregation network architecture.
- https://github.com/tshi04/ACCE for the abstraction-aggregation attention layer and loss function code
  - [aan_attention.py](./aan_attention.py)
- https://github.com/cgpotts/cs224u for some framework files, including
  - [sst.py](./sst.py)
  - [torch_model_base.py](./torch_model_base.py)
  - [torch_shallow_neural_classifier.py](./torch_shallow_neural_classifier.py)
  - [utils.py](./utils.py)
- https://github.com/cgpotts/dynasent for DynaSent models and data