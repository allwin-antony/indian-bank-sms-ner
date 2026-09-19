import json
import re
import os
import argparse
from typing import List, Dict, Tuple

# Taxonomy
LABELS = [
    'O',
    'B-TRANSACTION_AMOUNT', 'I-TRANSACTION_AMOUNT',
    'B-BALANCE_AMOUNT', 'I-BALANCE_AMOUNT',
    'B-MERCHANT', 'I-MERCHANT',
    'B-MERCHANT_VPA', 'I-MERCHANT_VPA',
    'B-ACCOUNT', 'I-ACCOUNT',
    'B-CARD', 'I-CARD',
    'B-REFERENCE_ID', 'I-REFERENCE_ID',
    'B-BANK_NAME', 'I-BANK_NAME',
    'B-TRANSACTION_DATE', 'I-TRANSACTION_DATE'
]

VALID_ENTITIES = {
    'TRANSACTION_AMOUNT', 'BALANCE_AMOUNT', 'MERCHANT', 'MERCHANT_VPA',
    'ACCOUNT', 'CARD', 'REFERENCE_ID', 'BANK_NAME', 'TRANSACTION_DATE'
}

ENTITY_ALIASES = {
    'TRANSACTION_TIME': 'TRANSACTION_DATE',
    'CURRENT_OUTSTANDING': 'BALANCE_AMOUNT',
    'TOTAL_DUE_AMOUNT': 'BALANCE_AMOUNT',
    'DUE_DATE': 'TRANSACTION_DATE',
}

def tokenize_with_spans(text: str) -> List[Tuple[str, int, int]]:
    """Tokenize text while keeping track of character spans for BIO alignment."""
    tokens = []
    for match in re.finditer(r'\w+|[^\w\s]', text):
        tokens.append((match.group(), match.start(), match.end()))
    return tokens

import difflib

def fuzzy_find(text: str, entity: str, threshold=0.8) -> Tuple[int, int]:
    """Find best fuzzy match of entity in text using sliding window."""
    best_ratio = 0
    best_span = (-1, -1)
    
    text_lower = text.lower()
    entity_lower = entity.lower()
    
    if len(entity_lower) == 0:
        return -1, -1
        
    window_sizes = [len(entity_lower) - 1, len(entity_lower), len(entity_lower) + 1, len(entity_lower) + 2]
    for w_size in window_sizes:
        if w_size <= 0: continue
        for i in range(len(text_lower) - w_size + 1):
            window = text_lower[i:i+w_size]
            ratio = difflib.SequenceMatcher(None, window, entity_lower).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_span = (i, i + w_size)
                
    if best_ratio >= threshold:
        return best_span
    return -1, -1

def find_entity_span(text: str, entity_value: str, entity_label: str) -> Tuple[int, int]:
    """Find the best matching span (start, end) for an entity in the text."""
    # 1. Exact match
    idx = text.find(entity_value)
    if idx != -1:
        return idx, idx + len(entity_value)
    
    # 2. Case-insensitive match
    lower_text = text.lower()
    lower_val = entity_value.lower()
    idx = lower_text.find(lower_val)
    if idx != -1:
        return idx, idx + len(entity_value)
        
    # 3. If numeric/amount, handle commas and decimal variations
    if 'AMOUNT' in entity_label:
        # e.g., value is 1000, text might have 1,000 or 1,000.00 or 1000.00
        clean_num = entity_value.replace(',', '')
        try:
            val_float = float(clean_num)
            # Search for numbers in text with possible commas/decimals
            for m in re.finditer(r'[\d,]+(?:\.\d+)?', text):
                m_str = m.group().replace(',', '')
                try:
                    if float(m_str) == val_float:
                        return m.start(), m.end()
                except ValueError:
                    pass
        except ValueError:
            pass

    # 4. If account or card, try searching for the digits/identifier
    if entity_label in ('ACCOUNT', 'CARD'):
        # Extract digits/suffix e.g. XX6251 or 6251
        digits_match = re.search(r'([A-Za-z*X]*\d{3,6})', entity_value)
        if digits_match:
            sub = digits_match.group(1)
            idx = text.find(sub)
            if idx != -1:
                return idx, idx + len(sub)
                
    # 5. Fuzzy Match fallback for slightly hallucinated entities
    f_start, f_end = fuzzy_find(text, entity_value, threshold=0.8)
    if f_start != -1:
        return f_start, f_end

    return -1, -1

def align_entities_to_bio(text: str, entities: Dict) -> Tuple[List[str], List[str]]:
    tokens_with_spans = tokenize_with_spans(text)
    tokens = [t[0] for t in tokens_with_spans]
    tags = ['O'] * len(tokens)
    
    if not isinstance(entities, dict):
        return tokens, tags

    # Clean and sort entities by value length descending to match longer phrases first
    cleaned_entities = []
    for k, v in entities.items():
        if v is None:
            continue
        v_str = str(v).strip()
        if not v_str:
            continue
        
        k_clean = k.strip().lstrip('_').upper()
        if k_clean in ENTITY_ALIASES:
            k_clean = ENTITY_ALIASES[k_clean]
            
        if k_clean in VALID_ENTITIES:
            cleaned_entities.append((k_clean, v_str))

    cleaned_entities.sort(key=lambda x: len(x[1]), reverse=True)

    for entity_label, entity_value in cleaned_entities:
        start_idx, end_idx = find_entity_span(text, entity_value, entity_label)
        if start_idx == -1:
            continue
            
        # Check if already tagged to avoid clobbering
        overlap = any(tags[i] != 'O' for i, (_, t_start, t_end) in enumerate(tokens_with_spans)
                      if t_start < end_idx and t_end > start_idx)
        if overlap:
            continue

        entity_started = False
        for i, (_, t_start, t_end) in enumerate(tokens_with_spans):
            if t_start >= start_idx and t_end <= end_idx:
                tags[i] = f'B-{entity_label}' if not entity_started else f'I-{entity_label}'
                entity_started = True
            elif t_start < end_idx and t_end > start_idx:
                tags[i] = f'B-{entity_label}' if not entity_started else f'I-{entity_label}'
                entity_started = True
                    
    return tokens, tags

def prepare_dataset(input_file: str, output_file: str):
    success_count = 0
    with open(input_file, 'r', encoding='utf-8') as fin, \
         open(output_file, 'w', encoding='utf-8') as fout:
        
        for line in fin:
            if not line.strip():
                continue
            data = json.loads(line)
            text = data.get('raw_body', data.get('body', ''))
            entities = data.get('entities', {})
            
            tokens, ner_tags = align_entities_to_bio(text, entities)
            
            if tokens:
                fout.write(json.dumps({
                    'id': data.get('id'),
                    'tokens': tokens,
                    'ner_tags': ner_tags,
                    'raw_body': text
                }) + '\n')
                success_count += 1
                
    print(f"Processed {success_count} samples and saved BIO tagged data to {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    
    prepare_dataset(args.input, args.output)
