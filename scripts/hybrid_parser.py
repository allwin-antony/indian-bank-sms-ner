import re
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForTokenClassification
import torch

class HybridSMSParser:
    def __init__(self, model_id="models/baseline"):
        print("Loading ONNX Model and Tokenizer for Hybrid Parsing...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        # We'll use CPUExecutionProvider explicitly
        self.model = ORTModelForTokenClassification.from_pretrained(model_id, provider="CPUExecutionProvider")
        
        # Regex Patterns
        self.amount_pattern = re.compile(r'(?i)(?:Rs\.?|INR)\s*([\d,]+\.?\d*)')
        self.account_pattern = re.compile(r'(?i)(?:A/c|card|account)\s*[Xx\*]*(\d{4})|XX(\d{4})|\*\*(\d{4})')
        self.balance_pattern = re.compile(r'(?i)(?:BAL|Avl\s*bal|Available\s*balance).*?(?:Rs\.?|INR)?\s*([\d,]+\.?\d*)')

    def parse_with_regex(self, sms: str) -> dict:
        extracted = {}
        
        # Extract Balance first
        bal_match = self.balance_pattern.search(sms)
        if bal_match:
            extracted['BALANCE'] = bal_match.group(1)
            
        # Extract Amount
        amounts = list(self.amount_pattern.finditer(sms))
        for amt in amounts:
            # Only count as AMOUNT if it doesn't overlap with BALANCE
            if bal_match and bal_match.start(1) <= amt.start(1) <= bal_match.end(1):
                continue
            extracted['AMOUNT'] = amt.group(1)
            break # Just take the first valid transaction amount
            
        # Extract Account
        acc_match = self.account_pattern.search(sms)
        if acc_match:
            # Find the non-None group
            for group in acc_match.groups():
                if group:
                    extracted['ACCOUNT'] = group
                    break
                    
        return extracted

    def parse_with_ner(self, sms: str) -> dict:
        inputs = self.tokenizer(sms, return_tensors="pt", truncation=True, max_length=128)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=2).squeeze().tolist()
        tokens = inputs.tokens()
        
        id2label = self.model.config.id2label
        
        extracted = {}
        current_entity = ""
        current_type = None
        
        for token, pred_id in zip(tokens, predictions):
            if token in ["[CLS]", "[SEP]"]:
                continue
                
            label = id2label[pred_id]
            
            if label.startswith("B-"):
                # Save previous entity if exists
                if current_entity and current_type:
                    # Clean subwords (e.g. "Sw" + "##iggy" -> "Swiggy")
                    clean_ent = current_entity.replace("##", "")
                    if current_type not in extracted:
                        extracted[current_type] = []
                    extracted[current_type].append(clean_ent.strip())
                    
                current_type = label[2:]
                current_entity = token
            elif label.startswith("I-") and current_type == label[2:]:
                if token.startswith("##"):
                    current_entity += token[2:]
                else:
                    current_entity += " " + token
            else:
                if current_entity and current_type:
                    clean_ent = current_entity.replace("##", "")
                    if current_type not in extracted:
                        extracted[current_type] = []
                    extracted[current_type].append(clean_ent.strip())
                current_entity = ""
                current_type = None
                
        # Handle trailing entity
        if current_entity and current_type:
            clean_ent = current_entity.replace("##", "")
            if current_type not in extracted:
                extracted[current_type] = []
            extracted[current_type].append(clean_ent.strip())
            
        return extracted
        
    def parse(self, sms: str) -> dict:
        result = {
            'AMOUNT': None,
            'BALANCE': None,
            'ACCOUNT': None,
            'MERCHANT': None,
            'TRANSACTION_TYPE': None,
            'PAY_MODE': None
        }
        
        # 1. Regex Pass (High confidence for numbers)
        regex_data = self.parse_with_regex(sms)
        result.update(regex_data)
        
        # 2. NER Pass (High confidence for unstructured text like merchants)
        ner_data = self.parse_with_ner(sms)
        
        # We only take MERCHANT, TRANSACTION_TYPE, and PAY_MODE from NER 
        # (or fallback to it if regex missed Amount/Account)
        for key in ['MERCHANT', 'TRANSACTION_TYPE', 'PAY_MODE']:
            if key in ner_data and ner_data[key]:
                result[key] = " ".join(ner_data[key])
                
        # Fallbacks just in case regex missed them
        if not result['AMOUNT'] and 'AMOUNT' in ner_data:
            result['AMOUNT'] = " ".join(ner_data['AMOUNT'])
            
        return result

if __name__ == "__main__":
    parser = HybridSMSParser()
    
    test_messages = [
        "Your a/c no. XX1234 is debited for Rs.500.00 on 14-02-23 and a/c linked to VPA ram@upi is credited (UPI Ref no 304512345678).",
        "Rs.2,500.00 credited to a/c XXXXXX4321 on 21-08-23 by A/c linked to mobile 9876543210 (IMPS Ref no 123456789012).",
        "Dear Customer, Rs.10,000.00 has been debited from your account **5678 on 01-09-23 at ATM. Available Balance is Rs.45,230.50.",
        "Payment of Rs.150.00 to Swiggy using UPI is successful. Your available balance is Rs.1,200.00."
    ]
    
    print("\n" + "="*50)
    print("TESTING HYBRID PARSER (Regex + NER)")
    print("="*50 + "\n")
    
    for msg in test_messages:
        print(f"SMS: {msg}")
        parsed = parser.parse(msg)
        for k, v in parsed.items():
            if v:
                print(f"  - {k}: {v}")
        print("-" * 50)
