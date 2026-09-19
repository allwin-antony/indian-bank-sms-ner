import json
import os
import argparse
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from seqeval.metrics import classification_report, f1_score

# Label mapping
LABELS = [
    'O',
    'B-AMOUNT', 'I-AMOUNT',
    'B-MERCHANT', 'I-MERCHANT',
    'B-ACCOUNT', 'I-ACCOUNT',
    'B-PAY_MODE', 'I-PAY_MODE',
    'B-BALANCE', 'I-BALANCE',
    'B-TRANSACTION_TYPE', 'I-TRANSACTION_TYPE'
]

LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for i, label in enumerate(LABELS)}

def load_data(file_path):
    tokens_list = []
    ner_tags_list = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            tokens = data['tokens']
            tags = data['ner_tags']
            
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
                # B-tag is odd index, I-tag is even index (B-AMOUNT=1, I-AMOUNT=2)
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

def train_model(input_file, output_dir, model_name="distilbert-base-uncased"):
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
        learning_rate=2e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
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
    
    print("Starting training...")
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
    
    # Export to ONNX
    print("\nExporting to ONNX...")
    try:
        import subprocess
        onnx_path = os.path.join(output_dir, "model.onnx")
        # Run optimum-cli to export to ONNX
        subprocess.run([
            "python3", "-m", "optimum.exporters.onnx",
            "--model", best_model_path,
            "--task", "token-classification",
            output_dir
        ], check=True)
        print(f"ONNX model saved in {output_dir}")
    except Exception as e:
        print(f"Failed to export to ONNX (you may need optimum library): {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Labeled JSONL file')
    parser.add_argument('--output', required=True, help='Output directory for model')
    args = parser.parse_args()
    
    train_model(args.input, args.output)
