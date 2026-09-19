import os
import json
import re
import argparse
import time
from dotenv import load_dotenv
from google import genai

# Load environment variables from .env file
load_dotenv()

# We will use Gemini 3.1 Flash Lite (15 RPM, 500 RPD)
MODEL = "gemini-3.1-flash-lite"
BATCH_SIZE = 20  # Increased batch size (but safe from max token cutoff)

BATCH_PROMPT_TEMPLATE = """You are a financial SMS annotator for Indian banks.
For each SMS below, output a JSON object with: intent, direction, entities.

Taxonomy:
- intent: TRANSACTION, REFUND, REVERSAL, FAILED, PENDING, BALANCE, BILL_REMINDER, PROMOTIONAL, OTP_SECURITY, SCAM, INFORMATIONAL
- direction: DEBIT, CREDIT, NONE
- entities (only include what is present): TRANSACTION_AMOUNT, BALANCE_AMOUNT, MERCHANT, MERCHANT_VPA, ACCOUNT, CARD, REFERENCE_ID, BANK_NAME, TRANSACTION_DATE

CRITICAL RULES:
1. Output ONLY a JSON array. No comments, no explanations, no trailing commas.
2. Each element in the array corresponds to the SMS at that index.

{sms_list}

Output a JSON array of {count} objects:"""


def extract_json(content):
    """Extract JSON from LLM output, handling markdown wrapping, comments, and trailing commas."""
    # Find all json blocks
    blocks = re.findall(r'```(?:json)?\n?(.*?)\n?```', content, re.DOTALL | re.IGNORECASE)
    if blocks:
        candidate = blocks[-1]
    else:
        candidate = content
        
    # Clean trailing commas which break Python's strict json.loads
    candidate = re.sub(r',\s*]', ']', candidate)
    candidate = re.sub(r',\s*}', '}', candidate)

    # Try to find a JSON array first
    arr_start = candidate.find('[')
    arr_end = candidate.rfind(']')
    if arr_start != -1 and arr_end != -1:
        try:
            return json.loads(candidate[arr_start:arr_end + 1])
        except json.JSONDecodeError:
            pass

    # Fall back to a single JSON object
    obj_start = candidate.find('{')
    obj_end = candidate.rfind('}')
    if obj_start != -1 and obj_end != -1:
        try:
            result = json.loads(candidate[obj_start:obj_end + 1])
            return [result] if isinstance(result, dict) else result
        except json.JSONDecodeError:
            pass

    return None


def process_batch(sms_batch, client):
    """Send a batch of SMS messages in a single API call and return parsed results."""
    sms_list_text = "\n".join(
        f"SMS[{i}]: \"{sms}\"" for i, sms in enumerate(sms_batch)
    )
    prompt = BATCH_PROMPT_TEMPLATE.replace("{sms_list}", sms_list_text).replace("{count}", str(len(sms_batch)))

    try:
        response = client.models.generate_content(
            model=MODEL,
            contents=prompt,
            config=genai.types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=8192,
                response_mime_type="application/json",
                safety_settings=[
                    genai.types.SafetySetting(
                        category=genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                        threshold=genai.types.HarmBlockThreshold.BLOCK_NONE
                    ),
                    genai.types.SafetySetting(
                        category=genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                        threshold=genai.types.HarmBlockThreshold.BLOCK_NONE
                    ),
                    genai.types.SafetySetting(
                        category=genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                        threshold=genai.types.HarmBlockThreshold.BLOCK_NONE
                    ),
                    genai.types.SafetySetting(
                        category=genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                        threshold=genai.types.HarmBlockThreshold.BLOCK_NONE
                    ),
                ]
            )
        )
        content = response.text
        if not content and response.candidates and response.candidates[0].content:
            parts = response.candidates[0].content.parts
            if parts:
                content = "".join([p.text for p in parts if p.text])
        
        if not content:
            print(f"  [WARN] API returned empty text. Reason: {response.candidates[0].finish_reason if response.candidates else 'Unknown'}")
            return [None] * len(sms_batch)

        parsed = extract_json(content)

        if parsed is None:
            print(f"  [WARN] Could not parse batch response. See data/error_batch.txt for raw output.")
            with open("data/error_batch.txt", "w", encoding="utf-8") as err_file:
                err_file.write(content)
            return [None] * len(sms_batch)

        # If the model returned fewer results than the batch, pad with None
        if isinstance(parsed, list):
            while len(parsed) < len(sms_batch):
                parsed.append(None)
            return parsed[:len(sms_batch)]
        else:
            return [None] * len(sms_batch)

    except Exception as e:
        print(f"  [ERROR] API request failed: {e}")
        return [None] * len(sms_batch)


def label_dataset(input_file, output_file, client):
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else '.', exist_ok=True)

    # Load input messages
    with open(input_file, 'r', encoding='utf-8') as fin:
        lines = fin.readlines()

    messages = []
    for i, line in enumerate(lines):
        try:
            data = json.loads(line)
            if 'raw_body' not in data and 'body' not in data and 'text' not in data:
                continue
            messages.append({
                'id': data.get('id', str(i)),
                'body': data.get('raw_body', data.get('body', data.get('text', '')))
            })
        except json.JSONDecodeError:
            continue

    # Resume support: load already-labeled IDs
    already_labeled = set()
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    d = json.loads(line)
                    already_labeled.add(d.get('id'))
                except json.JSONDecodeError:
                    continue
        print(f"Resuming: {len(already_labeled)} messages already labeled.")

    # Filter out already-labeled messages
    remaining = [m for m in messages if m['id'] not in already_labeled]
    total_batches = (len(remaining) + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"Loaded {len(messages)} total messages. {len(remaining)} remaining. {total_batches} batches of {BATCH_SIZE}.")

    success_count = len(already_labeled)

    for batch_idx in range(total_batches):
        start = batch_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(remaining))
        batch_msgs = remaining[start:end]
        batch_bodies = [m['body'] for m in batch_msgs]

        results = process_batch(batch_bodies, client)

        with open(output_file, 'a', encoding='utf-8') as fout:
            for msg, label_data in zip(batch_msgs, results):
                if label_data and isinstance(label_data, dict):
                    out_data = {
                        "id": msg['id'],
                        "raw_body": msg['body'],
                        "intent": label_data.get('intent', 'INFORMATIONAL'),
                        "direction": label_data.get('direction', 'NONE'),
                        "entities": label_data.get('entities', {})
                    }
                    fout.write(json.dumps(out_data) + '\n')
                    success_count += 1
                
        # Sleep to respect rate limits (30 RPM means 1 request per 2 seconds)
        time.sleep(2)

        print(f"Batch {batch_idx + 1}/{total_batches} done. Total labeled: {success_count}")

    print(f"\nFinished! Successfully labeled {success_count} out of {len(messages)} messages. Saved to {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='LLM Auto-label SMS for NER V2 (Batch Mode)')
    parser.add_argument('--input', required=True, help='Input JSONL file containing raw SMS')
    parser.add_argument('--output', required=True, help='Output tagged JSONL file')
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE, help=f'Number of SMS per API call (default: {BATCH_SIZE})')

    args = parser.parse_args()
    BATCH_SIZE = args.batch_size
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY environment variable not set. Please set it in your .env file or environment.")
        exit(1)

    client = genai.Client(api_key=api_key)
    label_dataset(args.input, args.output, client)
