import xml.etree.ElementTree as ET
import html
import json
import os
import argparse
import uuid

def extract_all_xml(file_paths, output_path):
    transactions = []
    
    for file_path in file_paths:
        print(f"Parsing {file_path}...")
        for event, elem in ET.iterparse(file_path, events=['end']):
            if elem.tag == 'sms':
                body = html.unescape(elem.get('body', ''))
                address = elem.get('address', '')
                
                transactions.append({
                    'id': str(uuid.uuid4()),
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
            
    print(f"Successfully extracted {len(transactions)} SMS to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract all SMS from XML')
    parser.add_argument('--input', nargs='+', required=True, help='Input XML files')
    parser.add_argument('--output', required=True, help='Output JSONL file')
    
    args = parser.parse_args()
    extract_all_xml(args.input, args.output)
