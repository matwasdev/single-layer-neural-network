# Single-Layer Neural Network for Multiclass Language Classification (Winner-Takes-All)

This project implements a classic single-layer neural network for multiclass classification, following the Winner-Takes-All (WTA) approach.
The key idea is that every language has a unique distribution of letter frequencies, and the model utilizes this characteristic for classification.

## Project Description
The model architecture:

* **Single layer** (no hidden layers).


* **One perceptron** per class — each perceptron is responsible for detecting one language.


* Each perceptron computes its own net value.


* The final predicted class is determined by selecting the perceptron with the maximum net output.


## Functionality
### Data Collection:
Load text files for multiple languages written in the Latin alphabet.

### Feature Extraction:
Generate a normalized vector of letter occurrences (26 letters A–Z).

### Dataset Handling:
Randomly shuffle and split the dataset into training and test sets (e.g., 80/20 split).

### Training:
Train each perceptron independently using the delta rule and update the threshold.
Initial weights are randomized and normalized.

### Classification:
For new inputs, compute net values for all perceptrons and assign the label of the perceptron with the highest net.

### Evaluation:
The model evaluates accuracy on both training and test dataset.


### User Interface:
Simple terminal interaction to classify custom input texts after training




## Example Dataset
Sample text data has been added to the text directory. 
These files contain text in multiple languages (English, Polish and Spanish). 
The program expects text files in this format to train and classify the neural network.



**Important:**
When classifying text manually,
the longer the text, the better the classification result, as it allows the model to better capture the letter distribution.
