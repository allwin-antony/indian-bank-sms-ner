# Indian Bank SMS NER & Intent Classifier

A highly optimized, edge-ready Natural Language Processing pipeline for extracting financial transactions and entities from messy Indian Banking SMS messages. This project is specifically tailored for deployment on Android applications via `onnxruntime-android`.

## 🧠 Architecture Overview

This project uses a dual-model approach based on `nreimers/MiniLM-L6-H384-uncased` to keep the memory footprint extremely low while maintaining high accuracy:

1. **Multi-Task Intent & Direction Classifier**: A custom sequence classification model that evaluates the entire SMS to determine the transaction intent (e.g., `TRANSACTION`, `BILL_PAYMENT`, `OTP`, `SPAM`) and the money flow direction (`CREDIT`, `DEBIT`, `NEUTRAL`).
2. **Token Classification (NER) Model**: An entity extraction model that reads the message token-by-token to extract complex, contextual entities like Bank Names, VPAs, Reference IDs, and Merchants.

### ⚡ The "Hybrid" Extraction Strategy
While Neural Networks are fantastic at understanding context, they can struggle with strict structured text (like amounts with commas, account numbers, and dates). To solve this, this project is designed to be paired with **Regex Post-Processing**. 

- **NER Model handles**: Contextual entities (`BANK_NAME`, `MERCHANT_VPA`, `REFERENCE_ID`, `MERCHANT`).
- **Regex Pipeline handles**: Structured entities (`TRANSACTION_AMOUNT`, `BALANCE_AMOUNT`, `ACCOUNT`, `TRANSACTION_DATE`).

By combining the 96%+ Intent classification and high-accuracy contextual NER with strict Regex patterns, we achieve a highly resilient parsing pipeline.

---

## 🛠️ Data Pipeline & Active Learning

Training robust NER models requires perfect datasets. Our initial dataset was annotated by `Qwen-2.5`, which produced noisy, misaligned labels. To fix this without breaking the bank on API costs, we implemented an **Active Learning Pipeline**:

1. **Baseline ONNX Scan**: We ran a baseline model over the 5,600 SMS dataset.
2. **Disagreement Filtering**: We isolated 1,697 "Hard Examples" where the baseline model disagreed with the Qwen labels.
3. **Gemini 3.1 Flash Lite**: We routed *only* the 1,697 hard examples through Gemini 3.1 Flash Lite using forced `application/json` mode.
4. **Fuzzy Matching Alignment**: Because LLMs often slightly hallucinate text during extraction (e.g., changing "Rs.1000" to "1000"), we wrote a custom Fuzzy String Matcher (`difflib.SequenceMatcher`) to perfectly re-align the LLM's extracted entities with the original SMS string indexes.
5. **The Gold Dataset**: The pristine labels were injected back into the dataset, producing a massive `gold_dataset.jsonl` with 5,603 perfectly annotated messages.

---

## 📊 Final Evaluation Metrics

After training on the Gold Dataset, our models achieved the following F1 scores:

### Intent & Direction Model
- **Intent Accuracy**: 96.3%
- **Intent F1**: 95.8%
- **Direction Accuracy**: 98.3%
- **Direction F1**: 98.4%

### Token Classification (NER) Model
*(Evaluated at 20 Epochs on purely neural extraction without Regex assist)*
- **BANK_NAME**: 94%
- **MERCHANT_VPA**: 94%
- **REFERENCE_ID**: 82%
- **MERCHANT**: 72%
- *Note: Amounts, accounts, and dates hover between 60-70% and should be handled by the Regex pipeline.*

---

## 📱 Android Export & INT8 Quantization

To run on an Android device without internet, Hugging Face PyTorch weights (`.safetensors`) are too heavy and slow. 

We provide an automated export script (`scripts/export_onnx.py`) that converts the models to the universal **ONNX format**.
To make the models even faster for mobile processors, the script automatically applies **INT8 Dynamic Quantization** to the NER model. This shrinks the NER model from ~90MB down to just **~22MB**, ensuring it loads instantly and saves battery life.

- **Quantized NER Model**: `models/onnx/ner/model_quantized.onnx` (22MB, INT8)
- **Intent Model**: `models/onnx/intent/model.onnx` (90MB, FP32 - Fallback due to custom shape constraints)

These models are plug-and-play ready for `onnxruntime-android`.

---

## 🚀 Usage Instructions

### 1. Prepare Data
Converts the JSONL data into BIO-tagged format and applies Fuzzy Matching to fix alignment hallucination.
```bash
python scripts/prepare_ner_data.py
```

### 2. Train Intent & Direction Model
Trains the multi-task classifier on the `gold_dataset`.
```bash
python scripts/train_intent.py --train_file data/train_gold.jsonl --val_file data/val_gold.jsonl --output_dir models/hf/intent --epochs 5
```

### 3. Train Token Classification (NER) Model
Trains the NER model. (Recommended: 15-20 epochs for token classification convergence).
```bash
python scripts/train_ner.py --input data/gold_ner.jsonl --output models/hf/ner
```

### 4. Export to Android (ONNX + INT8)
Exports both models to ONNX and quantizes the NER model to INT8 for mobile deployment.
```bash
python scripts/export_onnx.py
```
