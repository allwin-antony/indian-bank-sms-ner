import os
import json
import re
import argparse
import time
from dotenv import load_dotenv
from google import genai

load_dotenv()

MODEL = "gemini-3.1-flash-lite"
BATCH_SIZE = 20

PROMPT_TEMPLATE = """You are a strict QA system for financial SMS parsing.
Below are SMS messages along with their currently parsed entities.
Review them carefully. 
If the parsed entities are PERFECTLY CORRECT, output an empty object: {}
If there are errors (wrong amount, missed merchant, wrong type, etc), output ONLY the CORRECTED entities as a JSON object. Ensure amount is just a number (no Rs/INR).
Keys allowed in corrected object: amount, merchant, type, payment_mode, account_ref, category.

{sms_list}

Output MUST be a JSON array of length {count}. No explanations.
"""

def extract_json(content):
    blocks = re.findall(r'```(?:json)?\n?(.*?)\n?```', content, re.DOTALL | re.IGNORECASE)
    candidate = blocks[-1] if blocks else content
    candidate = re.sub(r',\s*]', ']', candidate)
    candidate = re.sub(r',\s*}', '}', candidate)

    arr_start = candidate.find('[')
    arr_end = candidate.rfind(']')
    if arr_start != -1 and arr_end != -1:
        try:
            return json.loads(candidate[arr_start:arr_end + 1])
        except:
            pass
    return None

def process_batch(batch, client):
    sms_list_text = ""
    for i, item in enumerate(batch):
        parsed_str = json.dumps(item.get('parsed', {})) if item.get('is_financial') else "Not Financial"
        sms_list_text += f"[{i}] SMS: \"{item['raw_body']}\"\nParsed: {parsed_str}\n\n"
    
    prompt = PROMPT_TEMPLATE.replace("{sms_list}", sms_list_text).replace("{count}", str(len(batch)))
    
    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json"
            )
        )
        content = response.text
        if not content: return [None] * len(batch)
        parsed = extract_json(content)
        if isinstance(parsed, list):
            while len(parsed) < len(batch): parsed.append(None)
            return parsed[:len(batch)]
    except Exception as e:
        print(f"API Error: {e}")
    return [None] * len(batch)

def validate_and_merge(input_file, output_file, client):
    with open(input_file, 'r', encoding='utf-8') as f:
        records = [json.loads(line) for line in f if line.strip()]
        
    print(f"Loaded {len(records)} records for validation.")
    
    # We only validate financial messages to save tokens, or we can validate all.
    # Let's validate only those that Dart thought were financial (to fix errors).
    # Optionally, we could validate non-financial to see if Dart missed them, 
    # but the user said "errors from this regex parser", usually referring to bad extractions.
    financial_records = [r for r in records if r.get('is_financial')]
    print(f"Validating {len(financial_records)} financial records...")
    
    corrections_made = 0
    total_batches = (len(financial_records) + BATCH_SIZE - 1) // BATCH_SIZE
    
    with open(output_file, 'w', encoding='utf-8') as fout:
        # We will write records out. To keep order, we could process in order.
        # Let's process the financial ones in batches.
        # But we need to output ALL records (even non-financial) to the final dataset.
        
        financial_idx = 0
        
        for r in records:
            if not r.get('is_financial'):
                # Just write it directly as not financial
                fout.write(json.dumps(r) + '\n')
                continue
                
            # Process a batch if needed
            # Wait, it's easier to just batch process all financial_records first,
            # store the results, then write everything.
            pass
            
    # Batch process
    for i in range(total_batches):
        batch = financial_records[i*BATCH_SIZE : (i+1)*BATCH_SIZE]
        results = process_batch(batch, client)
        
        for j, res in enumerate(results):
            record = batch[j]
            if res and isinstance(res, dict) and len(res) > 0:
                # LLM provided a correction!
                record['parsed'].update(res)
                record['llm_corrected'] = True
                corrections_made += 1
            else:
                record['llm_corrected'] = False
                
        print(f"Batch {i+1}/{total_batches} done. Corrections so far: {corrections_made}")
        time.sleep(2)
        
    # Now write everything
    with open(output_file, 'w', encoding='utf-8') as fout:
        for r in records:
            fout.write(json.dumps(r) + '\n')
            
    print(f"Finished. Total corrections made: {corrections_made}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    
    api_key = os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)
    
    validate_and_merge(args.input, args.output, client)
