import json
from transformers import AutoTokenizer, pipeline
from optimum.onnxruntime import ORTModelForTokenClassification

def main():
    model_id = "models/baseline"
    
    print("Loading tokenizer and ONNX model...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = ORTModelForTokenClassification.from_pretrained(model_id, provider="CPUExecutionProvider")
    
    # Create the NER pipeline
    ner_pipeline = pipeline(
        "token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple", # Groups B- and I- tokens into single entities
        device=-1
    )
    
    test_messages = [
        "Your a/c no. XX1234 is debited for Rs.500.00 on 14-02-23 and a/c linked to VPA ram@upi is credited (UPI Ref no 304512345678).",
        "Rs.2,500.00 credited to a/c XXXXXX4321 on 21-08-23 by A/c linked to mobile 9876543210 (IMPS Ref no 123456789012).",
        "Dear Customer, Rs.10,000.00 has been debited from your account **5678 on 01-09-23 at ATM. Available Balance is Rs.45,230.50.",
        "Payment of Rs.150.00 to Swiggy using UPI is successful. Your available balance is Rs.1,200.00."
    ]
    
    print("\n" + "="*50)
    print("TESTING NER MODEL ON SAMPLE SMS")
    print("="*50 + "\n")
    
    for msg in test_messages:
        print(f"SMS: {msg}")
        entities = ner_pipeline(msg)
        
        if not entities:
            print("Entities: NONE FOUND")
        else:
            print("Entities:")
            for ent in entities:
                print(f"  - {ent['entity_group']}: {ent['word']}")
        print("-" * 50)

if __name__ == "__main__":
    main()
