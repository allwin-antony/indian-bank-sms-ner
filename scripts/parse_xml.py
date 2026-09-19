import xml.etree.ElementTree as ET
import html
import re
import json
import os
import argparse

# Bank sender patterns to allow
ALLOWED_SENDERS = re.compile(
    r'(?i)(.*FEDBNK.*|.*HDFCBK.*|.*SBIUPI.*|.*PAYTMB.*|.*iPaytm.*|.*IPAYTM.*|.*BAJAJF.*|.*FLPKRT.*|.*AMAZON.*|.*AXIS.*|.*ICICI.*)'
)

# Transaction verbs
TX_KEYWORDS = [
    'debited', 'credited', 'transferred', 'received', 'withdrawn', 
    'deposited', 'reversed', 'sent', 'spent', 'paid', 'purchase'
]

# Amount pattern
AMOUNT_PATTERN = re.compile(r'(?:Rs\.?|INR)\s*[\d,]+\.?\d*', re.IGNORECASE)

# Rejection patterns (false positives)
REJECT_PATTERNS = [
    re.compile(r'(?i)OTP'),
    re.compile(r'(?i)verification code'),
    re.compile(r'(?i)is due on'),
    re.compile(r'(?i)keep ac funded'),
    re.compile(r'(?i)recharge of Rs'),
    re.compile(r'(?i)plan of Rs'),
    re.compile(r'(?i)will expire tomorrow')
]

def is_transaction(body: str, address: str) -> bool:
    if not ALLOWED_SENDERS.match(address):
        return False
    
    body_lower = body.lower()
    
    # Check for reject patterns
    for rp in REJECT_PATTERNS:
        if rp.search(body):
            return False
            
    # Check for transaction keywords
    has_keyword = any(kw in body_lower for kw in TX_KEYWORDS)
    if not has_keyword:
        return False
        
    # Check for amount
    if not AMOUNT_PATTERN.search(body):
        return False
        
    return True

def parse_xml_files(file_paths, output_path):
    transactions = []
    
    for file_path in file_paths:
        print(f"Parsing {file_path}...")
        for event, elem in ET.iterparse(file_path, events=['end']):
            if elem.tag == 'sms':
                body = html.unescape(elem.get('body', ''))
                address = elem.get('address', '')
                
                if is_transaction(body, address):
                    transactions.append({
                        'body': body,
                        'sender': address,
                        'date': elem.get('readable_date', ''),
                        'raw_date_ms': elem.get('date', '')
                    })
                elem.clear()
    
    # Save to jsonl
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        for tx in transactions:
            f.write(json.dumps(tx) + '\n')
            
    print(f"Successfully filtered and saved {len(transactions)} transaction SMS to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse XML and filter transaction SMS')
    parser.add_argument('--input', nargs='+', required=True, help='Input XML files')
    parser.add_argument('--output', required=True, help='Output JSONL file')
    
    args = parser.parse_args()
    parse_xml_files(args.input, args.output)
