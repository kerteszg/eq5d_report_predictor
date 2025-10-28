#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from transformers import AutoTokenizer, TFBertForSequenceClassification
from datasets import Dataset
import tensorflow as tf
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score
import os
import time


# # Multi-BERT Model Comparison: BERT, BioBERT, BlueBERT, SciBERT
# 
# This notebook compares four different BERT variants:
# - **BERT**: `bert-base-uncased` (general domain)
# - **BioBERT**: `dmis-lab/biobert-v1.1` (biomedical domain)
# - **BlueBERT**: `bionlp/bluebert_pubmed_uncased_L-24_H-1024_A-16` (biomedical domain)
# - **SciBERT**: `allenai/scibert_scivocab_uncased` (scientific domain)
# 
# ## Model Configurations

# In[ ]:


# Model configurations for all four BERT variants
BERT_MODELS = {
    'bert': {
        'name': 'BERT',
        'model_name': 'bert-base-uncased',
        'description': 'General domain BERT'
    },
    'biobert': {
        'name': 'BioBERT',
        'model_name': 'dmis-lab/biobert-v1.1',
        'description': 'Biomedical domain BERT'
    },
    'bluebert': {
        'name': 'BlueBERT',
        'model_name': 'bionlp/bluebert_pubmed_uncased_L-24_H-1024_A-16',
        'description': 'Biomedical domain BlueBERT'
    },
    'scibert': {
        'name': 'SciBERT',
        'model_name': 'allenai/scibert_scivocab_uncased',
        'description': 'Scientific domain BERT'
    }
}

def train_bert_model(lr, trainable, train_dataset, val_dataset, test_dataset, tokenizer, model_name, model_config):
    """
    Train BERT model for sequence classification.
    
    Args:
        lr: Learning rate
        trainable: Whether BERT layers are trainable
        train_dataset: Training dataset
        val_dataset: Validation dataset  
        test_dataset: Test dataset
        tokenizer: BERT tokenizer
        model_name: Name of the model variant
        model_config: Model configuration dictionary
        
    Returns:
        Dictionary containing training history length, evaluation metrics, and detailed metrics
    """
    print(f"\n{'='*60}")
    print(f"Training {model_config['name']} ({model_config['description']})")
    print(f"Model: {model_config['model_name']}")
    print(f"Learning rate: {lr}, Trainable: {trainable}")
    print(f"{'='*60}")
    
    # Load BERT model
    model = TFBertForSequenceClassification.from_pretrained(model_config['model_name'], from_pt=True)

    # Set up optimizer and loss function
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    # Configure model training
    model.layers[0].trainable = trainable
    model.summary()
    model.compile(optimizer=optimizer, loss=loss, metrics=['sparse_categorical_accuracy'])
    
    # Train model with early stopping
    es = tf.keras.callbacks.EarlyStopping(patience=10, monitor="val_loss", restore_best_weights=True)
    hist = model.fit(train_dataset, epochs=1000, 
            validation_data=val_dataset,
            callbacks=[es],
            verbose=1)
    
    # Evaluate on all datasets
    train_eval = model.evaluate(train_dataset, verbose=0)
    val_eval = model.evaluate(val_dataset, verbose=0)
    test_eval = model.evaluate(test_dataset, verbose=0)
    
    # Get predictions for detailed metrics calculation
    train_pred = model.predict(train_dataset, verbose=0)
    val_pred = model.predict(val_dataset, verbose=0)
    test_pred = model.predict(test_dataset, verbose=0)
    
    # Extract labels from datasets
    def extract_labels_from_dataset(dataset):
        labels = []
        for _, label in dataset:
            labels.extend(label.numpy().tolist())
        return np.array(labels)
    
    train_labels = extract_labels_from_dataset(train_dataset)
    val_labels = extract_labels_from_dataset(val_dataset)
    test_labels = extract_labels_from_dataset(test_dataset)
    
    # Get predictions
    train_pred_labels = np.argmax(train_pred.logits, axis=1)
    val_pred_labels = np.argmax(val_pred.logits, axis=1)
    test_pred_labels = np.argmax(test_pred.logits, axis=1)
    
    # Calculate detailed metrics for each dataset
    def calculate_metrics(y_true, y_pred):
        from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
        return {
            'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0),
            'accuracy': accuracy_score(y_true, y_pred)
        }
    
    train_metrics = calculate_metrics(train_labels, train_pred_labels)
    val_metrics = calculate_metrics(val_labels, val_pred_labels)
    test_metrics = calculate_metrics(test_labels, test_pred_labels)
    
    return {
        'model_name': model_name,
        'model_display_name': model_config['name'],
        'model_description': model_config['description'],
        'epochs_trained': len(hist.history["loss"]),
        'train_eval': train_eval,
        'val_eval': val_eval,
        'test_eval': test_eval,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics
    }


# In[ ]:


def load_and_preprocess_data(dataset_name, model_name):
    """
    Load and preprocess data for a specific dataset type and model.
    
    Args:
        dataset_name: One of 'title', 'title_abstract', 'abstract', 'title_abstract_keywords'
        model_name: One of 'bert', 'biobert', 'bluebert', 'scibert'
        
    Returns:
        Dictionary containing train_dataset, val_dataset, test_dataset, and tokenizer
    """
    model_config = BERT_MODELS[model_name]
    print(f"\n{'='*60}")
    print(f"Processing dataset: {dataset_name} for {model_config['name']}")
    print(f"{'='*60}")
    
    # Load training data
    df_train = pd.read_pickle(f"data/train_{dataset_name}.pkl")
    train_dataset = Dataset.from_pandas(df_train)
    
    # Fixed validation split for reproducible results
    _val_ids = [2, 7, 24, 32, 36, 47, 49, 59, 61, 71, 72, 86, 90, 95, 96]
    train_dataset = Dataset.from_pandas(df_train[~df_train.index.isin(_val_ids)])
    val_dataset = Dataset.from_pandas(df_train[df_train.index.isin(_val_ids)])
    
    # Load test data
    df_test = pd.read_pickle(f"data/test_{dataset_name}.pkl")
    test_dataset = Dataset.from_pandas(df_test)
    
    # Initialize tokenizer for the specific model
    tokenizer = AutoTokenizer.from_pretrained(model_config['model_name'])
    
    # Tokenize datasets
    train_encodings = tokenizer(train_dataset["text"], truncation=True, padding=True, max_length=512)
    val_encodings = tokenizer(val_dataset["text"], truncation=True, padding=True, max_length=512)
    test_encodings = tokenizer(test_dataset["text"], truncation=True, padding=True, max_length=512)
    
    # Extract labels
    train_labels = train_dataset["label"]
    val_labels = val_dataset["label"]
    test_labels = test_dataset["label"]
    
    # Create TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((
        dict(train_encodings),
        train_labels
    )).shuffle(100).batch(16)

    val_dataset = tf.data.Dataset.from_tensor_slices((
        dict(val_encodings),
        val_labels
    )).shuffle(100).batch(16)

    test_dataset = tf.data.Dataset.from_tensor_slices((
        dict(test_encodings),
        test_labels
    )).batch(16)
    
    print(f"Training samples: {len(train_labels)}")
    print(f"Validation samples: {len(val_labels)}")
    print(f"Test samples: {len(test_labels)}")
    print(f"Training positive ratio: {np.sum(train_labels) / len(train_labels):.3f}")
    print(f"Validation positive ratio: {np.sum(val_labels) / len(val_labels):.3f}")
    print(f"Test positive ratio: {np.sum(test_labels) / len(test_labels):.3f}")
    
    return {
        'train_dataset': train_dataset,
        'val_dataset': val_dataset,
        'test_dataset': test_dataset,
        'tokenizer': tokenizer
    }


# In[ ]:


# Main experiment loop for all models, datasets, and learning rates
dataset_types = ['title', 'title_abstract', 'abstract', 'title_abstract_keywords']
model_types = list(BERT_MODELS.keys())
learning_rates = [1e-4, 2e-4, 5e-4, 1e-5, 2e-5, 5e-5, 1e-6, 2e-6, 5e-6]
all_experiment_results = []

print(f"\n{'#'*100}")
print(f"COMPREHENSIVE MULTI-BERT EXPERIMENT")
print(f"Models: {[BERT_MODELS[m]['name'] for m in model_types]}")
print(f"Datasets: {dataset_types}")
print(f"Learning rates: {learning_rates}")
print(f"Total experiments: {len(model_types) * len(dataset_types) * len(learning_rates) * 5}")
print(f"{'#'*100}")

start_time = time.time()
experiment_count = 0
total_experiments = len(model_types) * len(dataset_types) * len(learning_rates) * 5

for model_name in model_types:
    model_config = BERT_MODELS[model_name]
    print(f"\n{'#'*100}")
    print(f"STARTING EXPERIMENTS FOR MODEL: {model_config['name'].upper()}")
    print(f"Model: {model_config['model_name']}")
    print(f"Description: {model_config['description']}")
    print(f"{'#'*100}")
    
    for dataset_type in dataset_types:
        print(f"\n{'='*80}")
        print(f"Processing dataset: {dataset_type.upper()} with {model_config['name']}")
        print(f"{'='*80}")
        
        # Load and preprocess data for this dataset type and model
        data = load_and_preprocess_data(dataset_type, model_name)
        train_dataset = data['train_dataset']
        val_dataset = data['val_dataset']
        test_dataset = data['test_dataset']
        tokenizer = data['tokenizer']
        
        # Train model with different learning rates for this dataset
        trainable = False  # Only Classifier Layer will be retrained
        dataset_results = []
        
        for lr in learning_rates:
            for iteration in range(5):
                experiment_count += 1
                elapsed_time = time.time() - start_time
                estimated_total = elapsed_time * total_experiments / experiment_count if experiment_count > 0 else 0
                remaining_time = estimated_total - elapsed_time
                
                print(f"\n--- Experiment {experiment_count}/{total_experiments} ---")
                print(f"Model: {model_config['name']}, Dataset: {dataset_type}, LR: {lr}, Iteration: {iteration}")
                print(f"Elapsed: {elapsed_time/3600:.2f}h, Estimated remaining: {remaining_time/3600:.2f}h")
                
                result = train_bert_model(lr, trainable, train_dataset, val_dataset, test_dataset, 
                                        tokenizer, model_name, model_config)
                
                # Store results with metadata
                result_entry = {
                    'model_name': model_name,
                    'model_display_name': model_config['name'],
                    'model_description': model_config['description'],
                    'dataset_type': dataset_type,
                    'learning_rate': lr,
                    'iteration': iteration,
                    'epochs_trained': result['epochs_trained'],
                    'train_metrics': result['train_metrics'],
                    'val_metrics': result['val_metrics'], 
                    'test_metrics': result['test_metrics']
                }
                dataset_results.append(result_entry)
                all_experiment_results.append(result_entry)
                
                print(f"Results - Accuracy: {result['test_metrics']['accuracy']:.4f}, "
                      f"F1: {result['test_metrics']['f1']:.4f}, "
                      f"Precision: {result['test_metrics']['precision']:.4f}, "
                      f"Recall: {result['test_metrics']['recall']:.4f}")
        
        print(f"\nCompleted {len(dataset_results)} experiments for {model_config['name']} on {dataset_type}")

total_time = time.time() - start_time
print(f"\n{'='*100}")
print(f"ALL EXPERIMENTS COMPLETED")
print(f"Total experiments: {len(all_experiment_results)}")
print(f"Total time: {total_time/3600:.2f} hours")
print(f"Average time per experiment: {total_time/len(all_experiment_results)/60:.2f} minutes")
print(f"{'='*100}")


# In[ ]:


# Process collected results for all models, datasets, and learning rates
import pandas as pd

# Convert results to DataFrame for easy analysis
results_df = pd.DataFrame(all_experiment_results)

# Extract metrics for easier analysis
def extract_metrics(metrics_dict):
    return pd.Series(metrics_dict)

# Create separate columns for each metric
test_metrics_df = results_df['test_metrics'].apply(extract_metrics)
val_metrics_df = results_df['val_metrics'].apply(extract_metrics)
train_metrics_df = results_df['train_metrics'].apply(extract_metrics)

# Add metric columns to main dataframe
for metric in ['accuracy', 'precision', 'recall', 'f1']:
    results_df[f'test_{metric}'] = test_metrics_df[metric]
    results_df[f'val_{metric}'] = val_metrics_df[metric]
    results_df[f'train_{metric}'] = train_metrics_df[metric]

# Display comprehensive summary statistics
print("="*100)
print("COMPREHENSIVE MULTI-BERT RESULTS SUMMARY")
print("="*100)
print(f"Total experiments: {len(results_df)}")
print(f"Models tested: {sorted(results_df['model_display_name'].unique())}")
print(f"Dataset types: {sorted(results_df['dataset_type'].unique())}")
print(f"Learning rates tested: {sorted(results_df['learning_rate'].unique())}")
print(f"Experiments per model: {results_df.groupby('model_display_name').size().iloc[0]}")

# Find best performing model overall
best_result = results_df.loc[results_df['test_accuracy'].idxmax()]
print(f"\nBEST PERFORMING MODEL OVERALL:")
print(f"Model: {best_result['model_display_name']} ({best_result['model_description']})")
print(f"Dataset: {best_result['dataset_type']}")
print(f"Learning rate: {best_result['learning_rate']}")
print(f"Iteration: {best_result['iteration']}")
print(f"Epochs trained: {best_result['epochs_trained']}")
print(f"Test metrics - Accuracy: {best_result['test_accuracy']:.4f}, "
      f"Precision: {best_result['test_precision']:.4f}, "
      f"Recall: {best_result['test_recall']:.4f}, "
      f"F1: {best_result['test_f1']:.4f}")

# Show results by model
print(f"\nRESULTS BY MODEL:")
model_summary = results_df.groupby('model_display_name')[['test_accuracy', 'test_precision', 'test_recall', 'test_f1']].agg(['mean', 'std', 'max']).round(4)
print(model_summary)

# Show results by dataset type
print(f"\nRESULTS BY DATASET TYPE:")
dataset_summary = results_df.groupby('dataset_type')[['test_accuracy', 'test_precision', 'test_recall', 'test_f1']].agg(['mean', 'std', 'max']).round(4)
print(dataset_summary)

# Show results by learning rate
print(f"\nRESULTS BY LEARNING RATE:")
lr_summary = results_df.groupby('learning_rate')[['test_accuracy', 'test_precision', 'test_recall', 'test_f1']].agg(['mean', 'std', 'max']).round(4)
print(lr_summary)

# Show best result for each model
print(f"\nBEST RESULT FOR EACH MODEL:")
for model_name in results_df['model_display_name'].unique():
    model_results = results_df[results_df['model_display_name'] == model_name]
    best_model_result = model_results.loc[model_results['test_accuracy'].idxmax()]
    print(f"{model_name}: Dataset={best_model_result['dataset_type']}, LR={best_model_result['learning_rate']}, "
          f"Acc={best_model_result['test_accuracy']:.4f}, F1={best_model_result['test_f1']:.4f}")

# Show best result for each dataset type
print(f"\nBEST RESULT FOR EACH DATASET TYPE:")
for dataset_type in results_df['dataset_type'].unique():
    dataset_results = results_df[results_df['dataset_type'] == dataset_type]
    best_dataset_result = dataset_results.loc[dataset_results['test_accuracy'].idxmax()]
    print(f"{dataset_type}: Model={best_dataset_result['model_display_name']}, LR={best_dataset_result['learning_rate']}, "
          f"Acc={best_dataset_result['test_accuracy']:.4f}, F1={best_dataset_result['test_f1']:.4f}")

# Show model comparison for each dataset
print(f"\nMODEL COMPARISON BY DATASET:")
for dataset_type in results_df['dataset_type'].unique():
    print(f"\n--- {dataset_type.upper()} ---")
    dataset_results = results_df[results_df['dataset_type'] == dataset_type]
    model_performance = dataset_results.groupby('model_display_name')['test_accuracy'].agg(['mean', 'std', 'max']).round(4)
    print(model_performance)

# Show detailed results for all experiments
print(f"\nDETAILED RESULTS FOR ALL EXPERIMENTS:")
display_cols = ['model_display_name', 'dataset_type', 'learning_rate', 'iteration', 'epochs_trained', 'test_accuracy', 'test_precision', 'test_recall', 'test_f1']
print(results_df[display_cols].round(4))

# Save results to CSV
results_df.to_csv('multi_bert_experiment_results.csv', index=False)
print(f"\nResults saved to 'multi_bert_experiment_results.csv'")

