# LLM-ERA
This repository contains the code for the paper **"Entropy Regularizing Activation: Boosting Continuous Control, Large Language Models, and Image Classification with Activation as Entropy Constraints"**([Link to Paper](TBD)). We introduce the ERA (Entropy Regularized Activation) function to enhance performance in reasoning tasks.

This codebase is adapted from [verl](https://github.com/volcengine/verl). Thanks for their great contributions.

## 📰Updates
- **2025.10.08**: We build the LLM-ERA codebase based on the original verl repository.

## 🎯Features
- Implementation of the ERA (Entropy Regularized Activation) function.
- Enhanced performance in reasoning tasks.

## 📖Installation
Simply run:
```bash
pip install -e .
```

## 🚀Usage
For a quick start, you can run the following command to train a Qwen-2.5-Math-7B model on the DAPO dataset:
```bash
bash ./qwen_7b_era_topk.sh
```
For evaluation, run
```
bash scripts/test.sh [YOUR_MODEL_PATH_HERE] [RESULT_PATH_HERE]
```
Make sure to adjust the parameters according to your setup and requirements. For more detailed options and configurations, refer to the verl documentation.
