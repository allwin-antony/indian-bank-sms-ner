import json
import re
import os
import argparse
from typing import List, Dict, Tuple

# Tokenizer pattern: split on whitespace and punctuation, but keep Rs.X intact as much as possible
# A simple approach for this NER task is to split by whitespace and specific punctuations
TOKENIZER_REGEX = re.compile(r'(?:Rs\.\s*[\d,]+\.?\d*|INR\s*[\d,]+\.?\d*|[\w]+|[^\w\s])')

def tokenize(text: str) -> List[str]:
    # Custom tokenizer that preserves amounts and handles punctuation well
    # First, let's normalize some spaces
    text = re.sub(r'Rs\.(\d)', r'Rs. \1', text)
    text = re.sub(r'BAL-Rs\.', 'BAL-Rs. ', text)
    
    # We will just split by whitespace and keep punctuation separate
    raw_tokens = re.findall(r"[\w]+|[^\w\s]", text)
    
    # Re-stitch amounts like Rs . 500 -> Rs.500 for simplicity? 
    # Actually WordPiece will subtokenize anyway. Let's just do a basic whitespace/punctuation split
    # A robust simple tokenizer:
    tokens = []
    current = ""
    for char in text:
        if char.isspace():
            if current:
                tokens.append(current)
                current = ""
        elif char.isalnum() or char in ['@', '*']: # keep VPA and masked accounts together
            current += char
        else:
            if current:
                if current.lower() in ['rs', 'inr', 'bal']:
                    # Special case, might be attached to punctuation
                    current += char
                else:
                    tokens.append(current)
                    tokens.append(char)
                    current = ""
            else:
                tokens.append(char)
                
    if current:
        tokens.append(current)
        
    return tokens

# Simplified regex-based auto labeler
def auto_label(sms_body: str) -> Tuple[List[str], List[str]]:
    # 1. Clean the body slightly
    body = sms_body.replace('\n', ' ')
    
    # 2. Extract spans using regex
    # Store tuples of (start_index, end_index, LABEL)
    spans = []
    
    # TRANSACTON_TYPE
    for match in re.finditer(r'(?i)\b(debited|credited|transferred|reversed|withdrawn|sent|paid)\b', body):
        spans.append((match.start(), match.end(), 'TRANSACTION_TYPE'))
        
    # AMOUNT (Transaction) - first amount near the start or before the type
    # We will find all amounts, and then assign the last one after BAL as BALANCE
    amounts = list(re.finditer(r'(?i)(?:Rs\.?|INR)\s*([\d,]+\.?\d*)', body))
    
    # Find BALANCE
    balance_match = None
    # Check "BAL-Rs.X" or "Avl bal INR" or "Available bal: INR"
    bal_match_1 = re.search(r'(?i)BAL\s*[-:]?\s*(?:Rs\.?|INR)\s*([\d,]+\.?\d*)', body)
    bal_match_2 = re.search(r'(?i)(?:Avl|Available)\s*bal(?:ance)?[:\s-]*(?:Rs\.?|INR)?\s*([\d,]+\.?\d*)', body)
    
    if bal_match_1:
        balance_match = bal_match_1
    elif bal_match_2:
        balance_match = bal_match_2
        
    if balance_match:
        spans.append((balance_match.start(1), balance_match.end(1), 'BALANCE'))
        
    # Any other amount not overlapping with balance is transaction AMOUNT
    for amt in amounts:
        # Check overlap
        is_balance = False
        if balance_match:
            if amt.start(1) >= balance_match.start() and amt.end(1) <= balance_match.end():
                is_balance = True
        
        if not is_balance:
            spans.append((amt.start(1), amt.end(1), 'AMOUNT'))
            
    # ACCOUNT
    for match in re.finditer(r'(?i)(?:A/c|card|account)\s*[Xx\*]*(\d{4})', body):
        spans.append((match.start(1), match.end(1), 'ACCOUNT'))
    for match in re.finditer(r'(?i)XX(\d{4})', body):
        spans.append((match.start(1), match.end(1), 'ACCOUNT'))
        
    # PAY_MODE
    for match in re.finditer(r'(?i)\b(UPI|ECOM Txn|POS Txn|NEFT|IMPS|ATM|card)\b', body):
        spans.append((match.start(), match.end(), 'PAY_MODE'))
        
    # MERCHANT (VPA or explicit)
    # VPA:
    for match in re.finditer(r'(?i)\b([a-z0-9\.\-]+@[a-z0-9]+)\b', body):
        spans.append((match.start(), match.end(), 'MERCHANT'))
        
    # Federal old style "at XYZ on"
    merch_old = re.search(r'(?i)at\s+([A-Z0-9\*\.\s]+?)\s+on\b', body)
    if merch_old:
        spans.append((merch_old.start(1), merch_old.end(1), 'MERCHANT'))
        
    # NEFT credit
    neft_cr = re.search(r'(?i)NEFT Cr-[A-Z0-9]+-([A-Z0-9\s]+)-', body)
    if neft_cr:
        spans.append((neft_cr.start(1), neft_cr.end(1), 'MERCHANT'))

    # 3. Tokenize
    # A very simple regex tokenizer
    tokens = re.findall(r"[\w'@*.-]+|[^\w\s]", body)
    
    # 4. Align tags
    tags = ['O'] * len(tokens)
    
    # Track character positions of tokens to align spans
    curr_pos = 0
    token_spans = []
    for token in tokens:
        start_idx = body.find(token, curr_pos)
        end_idx = start_idx + len(token)
        token_spans.append((start_idx, end_idx))
        curr_pos = end_idx

    # Assign BIO tags
    for s_start, s_end, label in spans:
        started = False
        for i, (t_start, t_end) in enumerate(token_spans):
            # If the token is inside the span or overlaps significantly
            if (t_start >= s_start and t_start < s_end) or (t_end > s_start and t_end <= s_end):
                if not started:
                    # Prevent overwriting a tag if already set, unless it's O
                    if tags[i] == 'O':
                        tags[i] = f'B-{label}'
                        started = True
                else:
                    if tags[i] == 'O':
                        tags[i] = f'I-{label}'
                        
    return tokens, tags

def label_dataset(input_file, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    processed = 0
    with open(input_file, 'r', encoding='utf-8') as fin, open(output_file, 'w', encoding='utf-8') as fout:
        for line in fin:
            data = json.loads(line)
            tokens, tags = auto_label(data['body'])
            
            # Skip if nothing was tagged (bad filter or missed by regex)
            if all(t == 'O' for t in tags):
                continue
                
            out_data = {
                'id': str(processed),
                'tokens': tokens,
                'ner_tags': tags,
                'raw_body': data['body']
            }
            fout.write(json.dumps(out_data) + '\n')
            processed += 1
            
    print(f"Auto-labeled {processed} messages and saved to {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Auto-label SMS for NER')
    parser.add_argument('--input', required=True, help='Filtered JSONL file')
    parser.add_argument('--output', required=True, help='Output tagged JSONL file')
    
    args = parser.parse_args()
    label_dataset(args.input, args.output)
