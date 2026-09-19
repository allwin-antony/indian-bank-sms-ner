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

### 5. Test Hybrid Inference (Recommended)
Since banking SMS are highly templated, we use a **Hybrid Approach**:
1. **Regex** for high-confidence structured fields (Amount, Account, Balance).
2. **NER Model** for unstructured fields (Merchant name, Transaction Type, Pay Mode).

You can test the hybrid parser with:
```bash
python scripts/hybrid_parser.py
```

## Model Performance & Architecture
- **Architecture**: `distilbert-base-uncased` fine-tuned for Token Classification.
- **Format**: Exported to `ONNX` (~265MB) for fast on-device inference (Android/iOS).
- **Training Data**: ~5,600 synthetically filtered and auto-labeled SMS messages based on regex heuristics from raw XML backups.
- **Subword Handling**: The model utilizes custom logic during training to correctly propagate labels to subwords (e.g. `de`, `##bit`, `##ed`), which the inference script reconstitutes back into full words.

### Evaluation Metrics
On a holdout set of ~850 messages, the model achieved a **99.5% F1-score**. 

*Classification Report Summary:*
| Entity | Precision | Recall | F1-Score |
|---|---|---|---|
| ACCOUNT | 1.00 | 1.00 | 1.00 |
| AMOUNT | 0.99 | 0.99 | 0.99 |
| BALANCE | 0.99 | 0.97 | 0.98 |
| MERCHANT | 0.99 | 0.99 | 0.99 |
| PAY_MODE | 1.00 | 1.00 | 1.00 |
| TRANSACTION_TYPE | 1.00 | 1.00 | 1.00 |

## Known Limitations & Roadmap
While the model performs exceptionally well on the synthetic dataset, third-party developers should be aware of the following:

1. **Merchant Coverage**: The current auto-labeler struggles to tag some un-prefixed merchants in the training data (e.g., catching `"to Swiggy"` without a clear VPA). As a result, the NER model occasionally misses simple names in production. Expanding the training regex will solve this.
2. **Duplicated Tags**: The hybrid script currently concatenates consecutive NER tags (e.g., `debited credited`), which needs to be deduped for final structured payloads.
3. **Hardware Acceleration**: The `hybrid_parser.py` explicitly uses `CPUExecutionProvider`. When porting to Android via ONNX Runtime, NNAPI delegates should be used for performance.
