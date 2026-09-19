import argparse
import os
import torch
import sys
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForTokenClassification
from onnxruntime.quantization import quantize_dynamic, QuantType

def export_to_onnx(model_id, output_dir, task):
    """
    Exports a PyTorch model to ONNX format and applies INT8 Dynamic Quantization for fast edge inference.
    """
    print(f"Loading {task} model from {model_id} and exporting to ONNX...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # We only use optimum for the standard HuggingFace NER model
    if task == "token-classification":
        model = ORTModelForTokenClassification.from_pretrained(model_id, export=True)
    else:
        raise ValueError("Unsupported task")

    # Save the FP32 ONNX model to disk
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Successfully exported FP32 {task} to {output_dir}")
    
    # Apply Dynamic INT8 Quantization
    print("Applying INT8 dynamic quantization...")
    onnx_fp32_path = os.path.join(output_dir, "model.onnx")
    onnx_int8_path = os.path.join(output_dir, "model_quantized.onnx")
    
    quantize_dynamic(
        model_input=onnx_fp32_path,
        model_output=onnx_int8_path,
        per_channel=True,
        reduce_range=True, # Recommended for x86/ARM to prevent overflow
        weight_type=QuantType.QInt8,
    )
    
    print(f"Successfully exported INT8 {task} to {onnx_int8_path}")


def export_custom_intent(model_id, output_dir):
    """
    Exports our custom MultiTaskIntentModel directly using torch.onnx and applies INT8 Quantization.
    """
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from train_intent import MultiTaskIntentModel
    
    print("Loading custom Intent model from PyTorch weights...")
    os.makedirs(output_dir, exist_ok=True)
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.save_pretrained(output_dir)
    
    model = MultiTaskIntentModel()
    model.load_state_dict(torch.load(os.path.join(model_id, "pytorch_model.bin"), map_location="cpu"))
    model.eval()
    
    # Create dummy inputs for traced export
    dummy_inputs = tokenizer("Your transaction of Rs. 100 has been debited", return_tensors="pt")
    input_ids = dummy_inputs["input_ids"]
    attention_mask = dummy_inputs["attention_mask"]
    
    onnx_fp32_path = os.path.join(output_dir, "model.onnx")
    onnx_int8_path = os.path.join(output_dir, "model_quantized.onnx")
    
    print("Exporting to FP32 ONNX...")
    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        onnx_fp32_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["intent_logits", "direction_logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "intent_logits": {0: "batch_size"},
            "direction_logits": {0: "batch_size"},
        },
        opset_version=14,
    )
    
    print("Applying INT8 dynamic quantization...")
    try:
        quantize_dynamic(
            model_input=onnx_fp32_path,
            model_output=onnx_int8_path,
            per_channel=True,
            reduce_range=True,
            weight_type=QuantType.QInt8,
        )
        print(f"Successfully exported INT8 Intent Model to {onnx_int8_path}")
    except Exception as e:
        print(f"INT8 Quantization failed for custom model (often due to shape inference on custom axes).")
        print(f"Falling back to FP32 model: {onnx_fp32_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--intent_model_dir", default="models/hf/intent", help="Path to trained intent model")
    parser.add_argument("--ner_model_dir", default="models/hf/ner/best_model", help="Path to trained NER model")
    parser.add_argument("--output_dir", default="models/onnx", help="Path to save ONNX models")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    ner_out = os.path.join(args.output_dir, "ner")
    export_to_onnx(args.ner_model_dir, ner_out, "token-classification")
    
    intent_out = os.path.join(args.output_dir, "intent")
    export_custom_intent(args.intent_model_dir, intent_out)
