# historical-syntax-bilstm-classifier
A NN classifier for identifying syntactic constructions in historical documents without parsing

## Data preprocessing
1) Run a coding query on a historical dataset (e.g., https://github.com/christopherahern/do-PPCHE)
2) Reformat the coding query (using reformat.q)
3) Modify convert-to-word to have DOCODES include a mapping from codingstring outputs to string integers
4) Run "python convert-to-word.py REFORMATED_OUTPUT" (this creates three conll files training, dev, and test)
5) Run "bilstm/bilstm.py" to train and test the model (outputs confusion matrices and classification reports)
