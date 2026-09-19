import json
import numpy as np
from transformers import AutoTokenizer
import onnxruntime as ort

def load_models():
    tokenizer = AutoTokenizer.from_pretrained("models/onnx/ner")
    intent_session = ort.InferenceSession("models/onnx/intent/model.onnx")
    return tokenizer, intent_session

INTENTS = ["TRANSACTION", "REFUND", "REVERSAL", "FAILED", "PENDING", "BALANCE", "BILL_REMINDER", "PROMOTIONAL", "OTP_SECURITY", "SCAM", "INFORMATIONAL"]
DIRS = ["DEBIT", "CREDIT", "NONE"]

def main():
    tokenizer, intent_session = load_models()
    
    input_file = "data/labeled_sms_qwen.jsonl"
    output_file = "data/disagreements.jsonl"
    
    print(f"Reading {input_file}...")
    with open(input_file, "r") as f:
        samples = [json.loads(line) for line in f]
        
    disagreements = []
    
    for i, s in enumerate(samples):
        text = s.get('raw_body', s.get('body', ''))
        llm_intent = s.get('intent', 'INFORMATIONAL')
        llm_dir = s.get('direction', 'NONE')
        
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        ort_inputs = {
            "input_ids": inputs["input_ids"].numpy(),
            "attention_mask": inputs["attention_mask"].numpy()
        }
        intent_logits, dir_logits = intent_session.run(None, ort_inputs)
        
        pred_intent = INTENTS[np.argmax(intent_logits[0])]
        pred_dir = DIRS[np.argmax(dir_logits[0])]
        
        if pred_intent != llm_intent or pred_dir != llm_dir:
            # Add to disagreements
            s['onnx_intent'] = pred_intent
            s['onnx_dir'] = pred_dir
            disagreements.append(s)
            
        if (i+1) % 500 == 0:
            print(f"Processed {i+1}/{len(samples)}... Found {len(disagreements)} disagreements.")

    print(f"\nFound {len(disagreements)} disagreements out of {len(samples)} total messages.")
    print(f"Saving to {output_file}...")
    
    with open(output_file, "w") as f:
        for d in disagreements:
            f.write(json.dumps(d) + "\n")
            
if __name__ == "__main__":
    main()
