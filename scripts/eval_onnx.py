import json
import numpy as np
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForTokenClassification
import onnxruntime as ort

def load_models():
    tokenizer = AutoTokenizer.from_pretrained("models/onnx/ner")
    ner_model = ORTModelForTokenClassification.from_pretrained("models/onnx/ner")
    intent_session = ort.InferenceSession("models/onnx/intent/model.onnx")
    return tokenizer, ner_model, intent_session

INTENTS = ["TRANSACTION", "REFUND", "REVERSAL", "FAILED", "PENDING", "BALANCE", "BILL_REMINDER", "PROMOTIONAL", "OTP_SECURITY", "SCAM", "INFORMATIONAL"]
DIRS = ["DEBIT", "CREDIT", "NONE"]

def predict(text, tokenizer, ner_model, intent_session):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    
    # 1. Intent
    ort_inputs = {
        "input_ids": inputs["input_ids"].numpy(),
        "attention_mask": inputs["attention_mask"].numpy()
    }
    intent_logits, dir_logits = intent_session.run(None, ort_inputs)
    
    intent_id = np.argmax(intent_logits[0])
    dir_id = np.argmax(dir_logits[0])
    
    pred_intent = INTENTS[intent_id]
    pred_dir = DIRS[dir_id]
    
    return pred_intent, pred_dir

def main():
    tokenizer, ner_model, intent_session = load_models()
    
    print("\nReading validation set to build a Gold Set evaluation...")
    with open("data/val_intent.jsonl", "r") as f:
        samples = [json.loads(line) for line in f]
        
    correct_intent = 0
    correct_dir = 0
    total = len(samples)
    
    disagreements = []
    
    for i, s in enumerate(samples):
        text = s.get('raw_body', s.get('body'))
        llm_intent = s.get('intent', 'INFORMATIONAL')
        llm_dir = s.get('direction', 'NONE')
        
        pred_intent, pred_dir = predict(text, tokenizer, ner_model, intent_session)
        
        if pred_intent == llm_intent:
            correct_intent += 1
        if pred_dir == llm_dir:
            correct_dir += 1
            
        if pred_intent != llm_intent or pred_dir != llm_dir:
            if len(disagreements) < 5:
                disagreements.append((text, llm_intent, llm_dir, pred_intent, pred_dir))

    print(f"\nTested on {total} validation samples (LLM Silver labels)")
    print(f"Intent Agreement with LLM:  {correct_intent/total:.2%}")
    print(f"Direction Agreement with LLM: {correct_dir/total:.2%}")
    
    print("\n--- Example Disagreements (Where Model disagrees with LLM) ---")
    for text, l_i, l_d, p_i, p_d in disagreements:
        print(f"\nTEXT: {text}")
        print(f"LLM Label (Silver):  {l_i} ({l_d})")
        print(f"ONNX Model Predicts: {p_i} ({p_d})")

if __name__ == "__main__":
    main()
