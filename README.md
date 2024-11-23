# Drug-Drug Interaction (DDI) Prediction Project

## Overview

This project predicts drug-drug interactions (DDI) using a machine learning-based approach. It utilizes **structured data** extracted from the **DDI Corpus**, a well-known dataset for drug interaction tasks. The extracted data is used to train a Random Forest classifier that predicts whether two drugs interact based on their types and the contextual information from the text.

---

## Dataset: DDI Corpus

The **DDI Corpus 2013** dataset is the foundation for this project. It provides:
1. **DrugBank data**: Includes expert-curated drug information.
2. **MedLine data**: Contains real-world medical literature.

### Extracted CSV Files
The DDI Corpus data has been preprocessed into the following CSV files:
1. **Entities CSV**: Contains drug-related entities extracted from the corpus.
   - Columns: `['doc_id', 'sent_id', 'entity_id', 'type', 'text', 'charOffset', 'source_type', 'file_name']`
2. **Pairs CSV**: Contains pairs of drugs with interaction labels (positive or negative).
   - Columns: `['doc_id', 'sent_id', 'pair_id', 'e1', 'e2', 'ddi', 'source_type', 'file_name']`
3. **Sentences CSV**: Provides the sentences in which drug pairs and entities occur.
   - Columns: `['doc_id', 'sent_id', 'text', 'source_type', 'file_name']`

### Data Statistics
- **Total Sentences**: 5,675
- **DDI Pairs**: 26,005
  - Positive Interactions: 3,789
  - Negative Interactions: 22,216
- **Entity Types**:
  - `Brand`: 1,423
  - `Drug`: 8,197
  - `Group`: 3,206
  - `Drug_n`: 103

---

## Features

- **Entity-Based Feature Engineering**:
  - Combines drug types, names, and context into a unified feature.
  - Adds type combination as a categorical feature.
- **Machine Learning Pipeline**:
  - Bag-of-Words with bigram extraction for text features.
  - Type combination encoding for entity interactions.
  - Random Forest classifier for binary interaction prediction.
- **Evaluation and Insights**:
  - Detailed classification metrics (precision, recall, F1-score).
  - Confusion matrix for error analysis.

---

## Requirements

### Python Dependencies
- **pandas**: Data manipulation.
- **numpy**: Numerical computation.
- **scikit-learn**: Machine learning model.
- **spaCy**: Natural language processing.

### Installation
Install the required packages:
```bash
pip install pandas numpy scikit-learn spacy
```

Download the **spaCy model**:
```bash
python -m spacy download en_core_web_sm
```

---

## Code Workflow

### 1. Data Preparation
The `prepare_data` function:
- Merges entities, sentences, and pairs data to create a training dataset.
- Generates new features:
  - **Combined Text**: Merges context, drug names, and types.
  - **Type Combination**: Encodes the interaction type pair (e.g., `Drug_Brand`).

### 2. Entity Analysis
The `analyze_entity_types` function:
- Explores the distribution of drug types and their most frequent combinations.

### 3. Model Training
The `train_model` function:
- Extracts text features using Bag-of-Words with bigram support.
- Encodes type combinations as categorical features.
- Trains a Random Forest classifier with balanced class weights to handle data imbalance.
- Evaluates the model on training and test data with metrics like precision, recall, and F1-score.

### 4. Prediction
The `predict_interaction` function:
- Takes two drugs, their types, and context as input.
- Predicts interaction (Yes/No) with a confidence score.

---

## Example Usage

### Running the Code
Run the project with:
```bash
python main.py
```

### Example Prediction
```python
result = predict_interaction(
    model, vectorizer, type_encoder,
    drug1="Aspirin", drug1_type="drug",
    drug2="Warfarin", drug2_type="drug",
    context="The patient is taking both medications for blood thinning."
)

print(result)
```

**Output**:
```plaintext
Drug 1: Aspirin (Type: drug)
Drug 2: Warfarin (Type: drug)
Interaction: Yes
Confidence: 87.5%
```

---

## Evaluation Metrics

The model outputs the following metrics:
1. **Precision, Recall, F1-Score**: To measure the balance between false positives and false negatives.
2. **Confusion Matrix**: For understanding classification performance in detail.

---

## Project Structure

```
.
├── entities_df.csv             # Extracted entities from DDI Corpus
├── sentences_df.csv            # Sentences from DDI Corpus
├── pairs_df.csv                # Drug-drug pairs with interaction labels
├── main.py                     # Main script for training and prediction
├── README.md                   # Project documentation
└── requirements.txt            # Dependency list
```

---

## Future Improvements

1. **Deep Learning Integration**: Use transformer-based models (e.g., BERT) for richer context understanding.
2. **Semantic Features**: Incorporate embeddings for drug names and types.
3. **Data Augmentation**: Address class imbalance through synthetic data generation.

---

## References

1. Segura-Bedmar, I., & others. (2023). **DDICorpus: Repository for Drug-Drug Interaction Dataset and Analysis**. Retrieved from [https://github.com/isegura/DDICorpus](https://github.com/isegura/DDICorpus)  
2. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. *Advances in Neural Information Processing Systems*, 30. [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)

---

## License

This project is open-source and licensed under the MIT License.
