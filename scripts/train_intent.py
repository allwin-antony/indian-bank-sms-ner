import os
import argparse
import json
import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments, EvalPrediction
from sklearn.metrics import f1_score, accuracy_score
import numpy as np

# Taxonomy
INTENT_LABELS = [
    "TRANSACTION", "REFUND", "REVERSAL", "FAILED", "PENDING", 
    "BALANCE", "BILL_REMINDER", "PROMOTIONAL", "OTP_SECURITY", 
    "SCAM", "INFORMATIONAL"
]
DIRECTION_LABELS = ["DEBIT", "CREDIT", "NONE"]

INTENT2ID = {label: i for i, label in enumerate(INTENT_LABELS)}
ID2INTENT = {i: label for label, i in INTENT2ID.items()}

DIR2ID = {label: i for i, label in enumerate(DIRECTION_LABELS)}
ID2DIR = {i: label for label, i in DIR2ID.items()}

class MultiTaskIntentModel(nn.Module):
    def __init__(self, model_name="nreimers/MiniLM-L6-H384-uncased"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size
        
        # Two classification heads
        self.intent_classifier = nn.Linear(hidden_size, len(INTENT_LABELS))
        self.direction_classifier = nn.Linear(hidden_size, len(DIRECTION_LABELS))
        
    def forward(self, input_ids, attention_mask, intent_labels=None, direction_labels=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :] 
        
        intent_logits = self.intent_classifier(pooled_output)
        direction_logits = self.direction_classifier(pooled_output)
        
        loss = None
        if intent_labels is not None and direction_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            intent_loss = loss_fct(intent_logits.view(-1, len(INTENT_LABELS)), intent_labels.view(-1))
            direction_loss = loss_fct(direction_logits.view(-1, len(DIRECTION_LABELS)), direction_labels.view(-1))
            # Equal weighting for now
            loss = intent_loss + direction_loss
            
        return {"loss": loss, "intent_logits": intent_logits, "direction_logits": direction_logits}

class IntentDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.texts = []
        self.intent_labels = []
        self.direction_labels = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                self.texts.append(data.get('raw_body', data.get('body', '')))
                intent_val = data.get('intent', 'INFORMATIONAL')
                if intent_val not in INTENT2ID:
                    intent_val = 'INFORMATIONAL'
                    
                dir_val = data.get('direction', 'NONE')
                if dir_val not in DIR2ID:
                    dir_val = 'NONE'
                    
                self.intent_labels.append(INTENT2ID[intent_val])
                self.direction_labels.append(DIR2ID[dir_val])
                
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "intent_labels": torch.tensor(self.intent_labels[idx], dtype=torch.long),
            "direction_labels": torch.tensor(self.direction_labels[idx], dtype=torch.long)
        }

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            intent_labels=inputs.get("intent_labels"),
            direction_labels=inputs.get("direction_labels")
        )
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred: EvalPrediction):
    # EvalPrediction contains a tuple of logits (intent, direction) and labels (intent, direction)
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    
    intent_logits = logits[0]
    direction_logits = logits[1]
    
    intent_labels = labels[0]
    direction_labels = labels[1]
    
    intent_preds = np.argmax(intent_logits, axis=-1)
    direction_preds = np.argmax(direction_logits, axis=-1)
    
    return {
        "intent_accuracy": accuracy_score(intent_labels, intent_preds),
        "intent_f1": f1_score(intent_labels, intent_preds, average="weighted"),
        "direction_accuracy": accuracy_score(direction_labels, direction_preds),
        "direction_f1": f1_score(direction_labels, direction_preds, average="weighted")
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--val_file", required=True)
    parser.add_argument("--output_dir", default="models/intent_model")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--model_name", default="nreimers/MiniLM-L6-H384-uncased")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = MultiTaskIntentModel(model_name=args.model_name)
    
    train_dataset = IntentDataset(args.train_file, tokenizer)
    val_dataset = IntentDataset(args.val_file, tokenizer)
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="intent_f1",
        remove_unused_columns=False # Required for custom forward arguments
    )
    
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )
    
    print(f"Starting training for Intent + Direction Classifier using {args.model_name}...")
    trainer.train()
    
    # Save the PyTorch model and tokenizer
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.output_dir, "pytorch_model.bin"))
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")

if __name__ == "__main__":
    main()
