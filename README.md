# Word Sense Disambiguation Repository

This repository contains a diverse array of foundational models designed to tackle Word Sense Disambiguation (WSD). The collection includes four baseline methodologies: Lesk, Naive Bayes, K-Nearest Neighbors, and BiLSTM. To enhance the comparative analysis, we have integrated a BERT-based model. The models were trained on the SemCor dataset and evaluated using both the Senseval and SemEval datasets, providing comprehensive performance assessment across distinct benchmarks.

## Directory Structure

```.
├── codes
│   ├── Baseline
│   │   ├── codeKNN_Naive.ipynb
│   │   └── codeLEST.ipynb
│   └── BERT-WSD
│       ├── code.ipynb
│       ├── codeLSTM.ipynb
│       ├── createFeatures.py
│       ├── datasetPreProcess.py
│       ├── modelBERT.py
│       └── training.py
├── Data
│   ├── knn_bert3.npy
│   ├── naive_bert_embeddings.npy
│   ├── semcor3.csv
│   └── semcor_copy.csv
├── Final_ReportNLP.pdf
├── NLP_ppt.pdf
├── README.md
└── Results
    ├── senseval2.gold.key.txt
    └── senseval2_predictions.txt
```
## Checkpoints
The BERT checkpoints are as follows:  
[Checkpoint-1000](https://drive.google.com/drive/folders/1-2FgXOB7RRynmdHkgenUxkTY5rImbECp?usp=sharing)

 [Checkpoint-2000](https://drive.google.com/drive/folders/101BHK7vlTERTvoO-4RRPJ-IFsqY7piuh?usp=sharing)


## Checkpoints

BERT checkpoints are available at the following links:

- [Checkpoint-1000](https://drive.google.com/drive/folders/1-2FgXOB7RRynmdHkgenUxkTY5rImbECp?usp=sharing)
- [Checkpoint-2000](https://drive.google.com/drive/folders/101BHK7vlTERTvoO-4RRPJ-IFsqY7piuh?usp=sharing)

## Running the Models

- For KNN, Naive Bayes, and Lesk algorithms, notebook files are provided. If you have the dataset available, run each block sequentially in the respective files.
- For the biLSTM model, refer to `codeLSTM.ipynb`.
- **BiLSTM Model Link**: [BiLSTM_Model](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/abhishek_shar_students_iiit_ac_in/EZKmiJlq6UdJrujB0gR2MBMBv8xuW-lRMWXlf5Rv8XYonw?e=9g5hH6)
- For the BERT-based model, a preprocessed dataset is included. If it's not present, execute `python3 datasetPreProcess.py`, making necessary edits to the XML and gold.key.txt files. After creating the dataset, generate features using `python3 createFeatures.py`. To initiate training, run `python3 training.py`, utilizing the model and tokenizer defined in `model.py`.

## Dataset

Download the dataset from [this link](https://iiitaphyd-my.sharepoint.com/:f:/g/personal/abhishek_shar_students_iiit_ac_in/EuY2tLElJ9RLmhpHkUs8zdMBE51Hetw8JfkMUR8UtL2vCg?e=5jyX9R).
