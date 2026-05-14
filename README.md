# Imitation Learning for CarRacing-v3

This project implements an imitation learning pipeline for the Gymnasium `CarRacing-v3` environment using:

- Behavioral Cloning
- DAgger (Dataset Aggregation)
- PyTorch
- ONNX export for submission compatibility

## Objective

The goal was to learn a driving policy from expert demonstrations. Behavioral Cloning was used first to imitate the expert, and DAgger was added to reduce covariate shift by collecting and relabeling states visited by the learner.

## Approach

The notebook includes:

- image preprocessing with crop, resize, grayscale, and normalization
- a CNN-based policy network
- supervised training with cross-entropy loss
- validation-based model selection
- iterative DAgger data aggregation and retraining
- final ONNX model export

## Results

The Behavioral Cloning policy achieved strong validation performance and a competitive driving score in evaluation. DAgger was then used to improve robustness by training on learner-visited states. Final models were evaluated in `CarRacing-v3` and exported in ONNX format for external testing.

## Files

- `expert.onnx`: expert policy used for DAgger
- `train/`: training demonstrations
- `val/`: validation demonstrations
- `model.onnx`: exported final policy

## Notes

The code is designed to run in Colab or locally with GPU support when available.
