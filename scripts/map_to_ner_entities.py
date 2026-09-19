import json
import argparse

def map_entities(input_file, output_file):
    success = 0
    with open(input_file, 'r', encoding='utf-8') as fin, open(output_file, 'w', encoding='utf-8') as fout:
        for line in fin:
            if not line.strip(): continue
            data = json.loads(line)
            
            if data.get('is_financial') and 'parsed' in data:
                parsed = data['parsed']
                entities = {}
                
                # Map extracted fields to NER taxonomy
                if 'amount' in parsed and parsed['amount'] is not None:
                    # Make sure amount is purely numerical
                    amount_str = str(parsed['amount'])
                    entities['TRANSACTION_AMOUNT'] = amount_str
                    
                if 'merchant' in parsed and parsed['merchant']:
                    entities['MERCHANT'] = parsed['merchant']
                    
                if 'account_ref' in parsed and parsed['account_ref']:
                    entities['ACCOUNT'] = parsed['account_ref']
                    
                data['entities'] = entities
                success += 1
            else:
                data['entities'] = {}
                
            fout.write(json.dumps(data) + '\n')
            
    print(f"Mapped {success} financial records to NER entities.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    map_entities(args.input, args.output)
