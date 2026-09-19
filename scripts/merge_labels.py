import json

def main():
    qwen_file = "data/labeled_sms_qwen.jsonl"
    corrected_file = "data/corrected_disagreements.jsonl"
    output_file = "data/gold_dataset.jsonl"
    
    # Load corrections into a dict by ID
    corrections = {}
    with open(corrected_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            corrections[data['id']] = data
            
    print(f"Loaded {len(corrections)} corrected messages.")
    
    # Merge
    merged_count = 0
    with open(qwen_file, 'r') as f_in, open(output_file, 'w') as f_out:
        for line in f_in:
            data = json.loads(line)
            msg_id = data.get('id')
            
            if msg_id in corrections:
                # Use the corrected data (which has the updated intent, direction, entities)
                f_out.write(json.dumps(corrections[msg_id]) + "\n")
                merged_count += 1
            else:
                # Use original Qwen data
                f_out.write(line)
                
    print(f"Merged {merged_count} corrections. Saved to {output_file}")

if __name__ == "__main__":
    main()
