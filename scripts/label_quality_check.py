import json
import argparse
from collections import Counter
import random

# ANSI Colors for terminal output
COLORS = {
    'B-AMOUNT': '\033[92m', # Green
    'I-AMOUNT': '\033[92m',
    'B-MERCHANT': '\033[94m', # Blue
    'I-MERCHANT': '\033[94m',
    'B-ACCOUNT': '\033[93m', # Yellow
    'I-ACCOUNT': '\033[93m',
    'B-PAY_MODE': '\033[96m', # Cyan
    'I-PAY_MODE': '\033[96m',
    'B-BALANCE': '\033[95m', # Magenta
    'I-BALANCE': '\033[95m',
    'B-TRANSACTION_TYPE': '\033[91m', # Red
    'I-TRANSACTION_TYPE': '\033[91m',
    'O': '\033[0m', # Reset
    'RESET': '\033[0m'
}

def check_quality(file_path):
    total = 0
    multi_amount_count = 0
    tag_counts = Counter()
    
    samples = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            total += 1
            
            tags = data['ner_tags']
            tokens = data['tokens']
            
            amt_count = sum(1 for t in tags if t == 'B-AMOUNT')
            if amt_count > 1:
                multi_amount_count += 1
                
            for tag in tags:
                if tag != 'O':
                    tag_counts[tag] += 1
                    
            if random.random() < 0.05 and len(samples) < 15:
                samples.append((tokens, tags, data.get('raw_body', '')))
                
    print(f"--- Quality Check Report ---")
    print(f"Total labeled SMS: {total}")
    print(f"Messages with multiple AMOUNT tags: {multi_amount_count}")
    print(f"\nTag distribution:")
    for k, v in tag_counts.most_common():
        print(f"  {k}: {v}")
        
    print(f"\n--- Random Samples ---")
    for tokens, tags, raw in samples:
        colored_text = []
        for t, tag in zip(tokens, tags):
            if tag != 'O':
                colored_text.append(f"{COLORS.get(tag, '')}{t}[{tag}]{COLORS['RESET']}")
            else:
                colored_text.append(t)
        
        print("RAW: " + raw.replace('\n', ' '))
        print("TAGGED: " + " ".join(colored_text))
        print("-" * 50)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Labeled JSONL file')
    args = parser.parse_args()
    check_quality(args.input)
