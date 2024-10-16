import mlflow
import mlflow.pytorch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import nltk
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.utils import resample
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from wordcloud import WordCloud
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from tqdm.notebook import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import os
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import spacy
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import nltk
import pandas as pd
import mlflow
import mlflow.pytorch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch.nn as nn
import torch.optim as optim
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import torch
# Initialize MLflow and create an experiment
mlflow.set_experiment("BERT")

# Load the dataset
df = pd.read_csv("pre_processed_data.csv")

# Tokenizer and Model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

# Define the function to create dataloader
def create_data_loader(df, tokenizer, max_len, batch_size):
    class SentimentDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_len):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_len = max_len

        def __len__(self):
            return len(self.texts)

        def __getitem__(self, item):
            text = str(self.texts[item])
            label = self.labels[item]
            encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_len,
                return_token_type_ids=False,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt',
            )
            return {
                'text': text,
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(label, dtype=torch.long)
            }

    ds = SentimentDataset(
        texts=df.content.to_numpy(),
        labels=df.label.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )
    return DataLoader(ds, batch_size=batch_size, num_workers=4)
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

# Define a function to categorize ratings into sentiment classes
def categorize_sentiment(rating):
    if rating in [4, 5]:
        return 'Positive'
    elif rating == 3:
        return 'Neutral'
    else:
        return 'Negative'

# Apply the function to create a new 'sentiment' column
df['sentiment'] = df['rating'].apply(categorize_sentiment)

# Map sentiments to numeric labels
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['sentiment'])

# Ensure you have the 'label' column by this point
print(df.head())  # Check that 'label' is present
df_train, df_val = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
train_data_loader = create_data_loader(df_train, tokenizer, max_len=160, batch_size=16)
val_data_loader = create_data_loader(df_val, tokenizer, max_len=160, batch_size=16)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch"
)

# Compute metrics
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Wrap the training in an MLFlow run
with mlflow.start_run(run_name="BERT Sentiment Analysis"):

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data_loader.dataset,
        eval_dataset=val_data_loader.dataset,
        compute_metrics=compute_metrics
    )

    # Log the model architecture
    mlflow.pytorch.log_model(model, "bert_model")

    # Train the model
    trainer.train()

    # Evaluate the model
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")

    # Log metrics
    mlflow.log_metrics({
        "accuracy": eval_results["eval_accuracy"],
        "precision": eval_results["eval_precision"],
        "recall": eval_results["eval_recall"],
        "f1": eval_results["eval_f1"]
    })

# Save the model
model.save_pretrained("bert_sentiment_model")
tokenizer.save_pretrained("bert_sentiment_tokenizer")
