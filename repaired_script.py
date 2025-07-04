import argparse
import random
import re
import sys
from tqdm import tqdm
 
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
 
 
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate arithmetic accuracy for various prompt styles using greedy decoding and seed variation."
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="meta-llama/Meta-Llama-3-8B",
        help="HuggingFace model identifier (default: Meta-Llama-3-8B)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=1000,
        help="Number of random (A,B) examples per style/op per seed",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs='+',
        default=[42, 123, 999],
        help="List of integer seeds for reproducibility (default: 42 123 999)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on",
    )
    args = parser.parse_args()
 
    # load model & tokenizer
    print(f"Loading model {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if "cuda" in args.device else torch.float32
    ).to(args.device)
    print(f"Model loaded successfully")
    model.eval()
 
    # exactly the 12 templates from Table 4 (including trailing spaces)
    templates = {
        "Original": {
            "A+B": "{A}+{B}=",
            "A-B": "{A}-{B}=",
        },
        "Verbal Math": {
            "A+B": "The sum of {A} and {B} is ",
            "A-B": "The difference of {A} and {B} is ",
        },
        "Question Answering": {
            "A+B": "What is the sum of {A} and {B}? Answer: ",
            "A-B": "What is the difference of {A} and {B}? Answer: ",
        },
        "Instruction": {
            "A+B": "If you add {A} to {B}, you will get ",
            "A-B": "If you subtract {B} from {A}, you will get ",
        },
        "Math word Problem": {
            "A+B": "John has {A} cookies. Jane has {B} cookies. Together they have ",
            "A-B": "John has {A} cookies. He gave Jane {B} cookies. John now has ",
        },
        "Python Program": {
            "A+B": "a = {A}; b = {B}; print(a + b) # should print ",
            "A-B": "a = {A}; b = {B}; print(a - b) # should print ",
        },
    }
 
    # initialize structure to collect accuracies per seed
    results = {style: {op: [] for op in ops} for style, ops in templates.items()}
 
    for seed in args.seeds:
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
 
        print(f"\nEvaluating seed {seed}...")
        for style, ops in templates.items():
            if style != "Original":
                continue
            for op_symbol, template in ops.items():
                correct = 0
                total = args.num_samples
                desc = f"Seed {seed} | {style:>20} | {op_symbol}"
 
                for _ in tqdm(range(total), desc=desc, file=sys.stdout):
                    # sample A,B such that result in [0,999]
                    while True:
                        A = random.randint(0, 100)
                        B = random.randint(0, 100)
                        answer = A + B if op_symbol == "A+B" else A - B
                        if 0 <= answer <= 999:
                            break
 
                    prompt = template.format(A=A, B=B)
                    inputs = tokenizer(prompt, return_tensors="pt").to(args.device)
                    # print each token in the prompt in text
                    # print(f"Prompt: {prompt}")
                    # print(f"Tokens: {tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])}")
                    # greedy decoding
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=5,
                        do_sample=False,
                        num_beams=1,
                        early_stopping=True,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                    gen_tokens = outputs[0, inputs["input_ids"].shape[-1]:]
                    generated = tokenizer.decode(gen_tokens, skip_special_tokens=True)
 
                    m = re.match(r"-?\d+", generated.strip())
                    pred = m.group(0) if m else ""
 
                    if pred == str(answer):
                        correct += 1
 
                acc = correct / total
                results[style][op_symbol].append(acc)
 
    # aggregate statistics
    print("\nAggregated Accuracies over seeds:", args.seeds)
    print(f"{'Style':<20} {'Op':<5} {'Min':>8} {'Avg':>8} {'Max':>8}")
    print("-" * 55)
    for style, ops in results.items():
        for op_symbol, acc_list in ops.items():
            min_acc = min(acc_list)
            max_acc = max(acc_list)
            avg_acc = sum(acc_list) / len(acc_list)
            print(f"{style:<20} {op_symbol:<5} {min_acc:8.3f} {avg_acc:8.3f} {max_acc:8.3f}")
 
 
if __name__ == "__main__":
    main()