This directory contains the code necessary to evaluate a result json structure from the Results folder.
- Only `evaluate_individual_scores.py` contains directly runnable code. This can be run directly with additional arguments
specifying the metric and structure to evaluate. Note that BLEU should not be used unless for ordering individual
sentences, since this uses an implementation other than the `multi-bleu.perl` elsewhere in the code. 
This means those results can not be compared.
