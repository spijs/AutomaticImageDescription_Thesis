# thesisImageNL
### What is this?
This repository contains the code accompanying a master's thesis "Describing Images Using Natural Language".
The code is an extension of [NeuralTalk](https://github.com/karpathy/neuraltalk). It contains several RNN-based language models.

The root of this repository contains some scripts to create a perturbed version of training data, to learn topic models, to calculate idf weights, ... For mor info, please do check the comments in the individual files.

### How to run? 
To train a neural network, run __driver.py__. To evaluate the results of training, run __eval_sentence_predictions.py__.
Each runnable file contains a help function and extensive comments explaining how it works.
