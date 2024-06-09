# Biomedical Event Trigger Detection

## Table of Contents
- [Project Overview](#project-overview)
  - [Project Background](#project-background)
  - [Key Features](#key-features)
  - [Technology Stack](#technology-stack)
- [Environment Requirements](#environment-requirements)
  - [Python Version](#python-version)
  - [Dependencies](#dependencies)
- [Quick Start](#quick-start)
  - [Configuration File](#configuration-file)
    - [How to Edit the Configuration File](#how-to-edit-the-configuration-file)
  - [Data Preparation](#data-preparation)
    - [Dataset Source](#dataset-source)
    - [Contextual Data](#contextual-data)
  - [Training the Model](#training-the-model)

## Project Overview

### Project Background

Biomedical event trigger detection aims at identifying specific words or phrases within a text that indicate the presence of biological events or processes. This task has received extensive research attention in the past ten years due to its significance in understanding biomedical literature and facilitating tasks such as information extraction and knowledge discovery. Our project leverages advanced neural network models to improve the accuracy and robustness of trigger detection by incorporating external knowledge sources.

### Key Features

- **Biomedical Event Trigger Detection**: Utilizes state-of-the-art neural network models to identify event triggers in biomedical texts.
- **Contextual Integration**: Incorporates external contextual information from sources like Wikipedia and PubMed to enhance model performance.
- **Comprehensive Evaluation**: Includes thorough experimental evaluation and error analysis to understand the model's capabilities and limitations.

### Technology Stack

- **PyTorch**: Used for model building and training.
- **Transformers**: Employed for implementing BERT-based models.
- **CRF**: Conditional Random Fields used for sequence labeling.
- **AdamW**: Optimizer used for training the models.
- **OneCycleLR**: Learning rate scheduling strategy used during training.

## Environment Requirements

### Python Version

This project is implemented using Python 3.7.

### Dependencies

- PyTorch 1.13.1
- Transformers
- tqdm
- sklearn
- numpy

Additional package dependencies can be found in the `requirements.txt` file and installed using pip:

```bash
pip install -r requirements.txt
```

## Quick Start

### Configuration File

The configuration for the model, optimizer, and other settings can be found in the YAML file (`config.yaml`). Make sure to go through the configuration to understand the different parameters.

#### How to Edit the Configuration File

To edit the configuration file, simply open `config.yaml` in a text editor and modify the values according to your needs. For example, to change the learning rate:

```yaml
train:
  learning_rate: 2e-5  # Change this value
```
## Data Preparation
###Dataset Source
The dataset used for this project is the MLEE dataset. The original dataset can be downloaded from here. The data is annotated in BIO format for sequence labeling tasks.

### Contextual Data
In this project, contextual information is added from external sources such as Wikipedia and PubMed. Detailed steps on how to preprocess and integrate this contextual data are provided in the GitHub repository.

## Training the Model
### Run the training script:

```bash
python train.py
```
