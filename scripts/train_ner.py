import json
import os
import argparse
import numpy as np
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from seqeval.metrics import classification_report, f1_score

# New V2 Label mapping
LABELS = [
    'O',
    'B-TRANSACTION_AMOUNT', 'I-TRANSACTION_AMOUNT',
    'B-BALANCE_AMOUNT', 'I-BALANCE_AMOUNT',
    'B-MERCHANT', 'I-MERCHANT',
    'B-MERCHANT_VPA', 'I-MERCHANT_VPA',
    'B-ACCOUNT', 'I-ACCOUNT',
    'B-CARD', 'I-CARD',
    'B-REFERENCE_ID', 'I-REFERENCE_ID',
    'B-BANK_NAME', 'I-BANK_NAME',
    'B-TRANSACTION_DATE', 'I-TRANSACTION_DATE'
]

LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for i, label in enumerate(LABELS)}

def load_data(file_path="data/gold_ner.jsonl"):
    tokens_list = []
    ner_tags_list = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            tokens = data.get('tokens', [])
            tags = data.get('ner_tags', [])
            
            if not tokens or not tags:
                continue
                
            # Map string tags to IDs
            tag_ids = [LABEL2ID.get(t, 0) for t in tags]
            
            tokens_list.append(tokens)
            ner_tags_list.append(tag_ids)
            
    return Dataset.from_dict({
        'tokens': tokens_list,
        'ner_tags': ner_tags_list
    })

def tokenize_and_align_labels(examples, tokenizer):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True, max_length=128
    )

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                # Propagate label to continuation subwords
                parent_label = label[word_idx]
                if parent_label % 2 == 1:
                    label_ids.append(parent_label + 1)
                else:
                    label_ids.append(parent_label)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [ID2LABEL[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [ID2LABEL[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    f1 = f1_score(true_labels, true_predictions)
    return {"f1": f1}

def train_model(input_file, output_dir, model_name="nreimers/MiniLM-L6-H384-uncased"):
    print(f"Loading data from {input_file}...")
    dataset = load_data(input_file)
    
    # Split into train and eval
    dataset = dataset.train_test_split(test_size=0.15, seed=42)
    print(f"Train size: {len(dataset['train'])}, Eval size: {len(dataset['test'])}")
    
    print(f"Loading tokenizer and model ({model_name})...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )
    
    tokenized_datasets = dataset.map(
        lambda x: tokenize_and_align_labels(x, tokenizer),
        batched=True
    )
    
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=3e-5, # Slightly higher learning rate for MiniLM
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=20,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    print(f"Starting NER training with {model_name}...")
    trainer.train()
    
    # Save the final model
    best_model_path = os.path.join(output_dir, "best_model")
    trainer.save_model(best_model_path)
    print(f"Model saved to {best_model_path}")
    
    # Evaluate and print full report
    print("\nEvaluating best model...")
    predictions, labels, _ = trainer.predict(tokenized_datasets["test"])
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [ID2LABEL[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [ID2LABEL[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    
    report = classification_report(true_labels, true_predictions)
    print("\n--- Classification Report ---")
    print(report)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Labeled JSONL file for NER')
    parser.add_argument('--output', required=True, help='Output directory for model')
    parser.add_argument('--model_name', default="nreimers/MiniLM-L6-H384-uncased")
    args = parser.parse_args()
    
    train_model(args.input, args.output, args.model_name)
