# Indian Bank SMS NER (Token Classifier)

This project builds a lightweight Token Classification (NER) model to parse Indian banking SMS messages and extract structured transaction details. The model is trained on a synthetic dataset generated from real XML backups using regex patterns and exported to ONNX format for on-device inference (e.g. Android apps).

## Entities Extracted

* `AMOUNT`: The transaction value
* `MERCHANT`: The payee name or UPI VPA (e.g. `upiswiggy@icici`)
* `ACCOUNT`: Masked account numbers (e.g. `XX1234`)
* `PAY_MODE`: Transaction mode (`UPI`, `NEFT`, `IMPS`, `POS Txn`, etc.)
* `BALANCE`: The available balance left in the account
* `TRANSACTION_TYPE`: The direction of flow (`debited`, `credited`, `reversed`, etc.)

## Usage

### 1. Setup

```bash
pip install -r requirements.txt
pip install "optimum[onnxruntime]"
```

### 2. Prepare Data
Place your SMS Backup & Restore XML files in `data/raw/`.
Run the data filtering and auto-labeling pipeline:

```bash
python scripts/parse_xml.py --input data/raw/sms*.xml --output data/processed/filtered_transactions.jsonl
python scripts/auto_label.py --input data/processed/filtered_transactions.jsonl --output data/processed/bio_tagged_dataset.jsonl
```

### 3. Verify Labels
You can visually verify the generated BIO tags:
```bash
python scripts/label_quality_check.py --input data/processed/bio_tagged_dataset.jsonl
```

### 4. Train Model
Run the Hugging Face training script:
```bash
python scripts/train.py --input data/processed/bio_tagged_dataset.jsonl --output models/baseline/
```
The resulting ONNX model will be saved inside `models/baseline/`.

### 5. Test Inference
You can test the exported ONNX model against sample messages to verify it extracts entities correctly.
```bash
python scripts/test_inference.py
```
