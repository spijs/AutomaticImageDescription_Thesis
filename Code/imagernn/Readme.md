The code is organized as follows:

- `data_provider.py` abstracts away the datasets and provides uniform API for the code.
- `utils.py` is what it sounds like it is :)
- `solver.py`: the solver class doesn't know anything about images or sentences, it gets a model and the gradients and performs a step update
- `generic_batch_generator.py` handles batching across a batch of image/sentences that need to be forwarded through the networks. It calls the
- `lstm_generator.py`, which is an implementation of the Google LSTM for generating images.
- `imagernn_utils.py` contains some image-rnn specific utilities, such as evaluation function etc. These come in handy when we want to use some functionality across different scripts (e.g. driver and evaluator)
- `rnn_generator.py` has a simple RNN implementation for now, an alternative to LSTM.
- `fsmn_generator.py` has a basic FSMN implementation.
- `Hlayer.py` is the hidden layer implementation of the Hlayer.
- `glstm_generator.py` is the gLSTM implementation which is an extension of lstm_generator.
- `guide.py` abstracts away everything necessary to retrieve the guide information.
- `get_sentence_stats.py` is used to retrieve mean and standard deviation statistics of a training set.