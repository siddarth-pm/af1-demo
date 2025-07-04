"""
This script runs an experiment to test the AF1 circuit on Llama 3 8B model.

Only contains code for the "Original" template.

The experiment is configured to:
1.  Load a pre-trained transformer model (e.g., Llama 3, Pythia).
2.  Generate a dataset of arithmetic problems (e.g., "a+b=").
3.  Load in pre-calculated CAMA representations of the second and third operands.
4.  Run the full AF1 circuit.
5.  Measure the model's *recovered* accuracy on the arithmetic task under these conditions
    ("AF1 faith") and compare it to its base accuracy.
6.  The final results are printed in a table.
"""
from tqdm import tqdm
import transformer_lens as lens
import torch
import random
import pickle
import os
import logging
import argparse
from typing import List, Tuple, Dict, Any, Callable, cast
random.seed(42)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Imports complete.")



def perform_waiting_experiment(
    model: lens.HookedTransformer,
    second_waiting_toks: List[int],
    third_waiting_toks: List[int],
    second_representations: Dict[int, Any],
    third_representations: Dict[int, Any],
    second_operand_pos: int,
    second_waiting_offset: int,
    third_waiting_offset: int,
    eval_examples: List[Tuple[str, str]],
    test_config: Dict[str, Any],
    batch_size: int = 20,
) -> float:
    """
    Performs the main waiting experiment using CAMA and attention pattern blocking.

    Args:
        model: The HookedTransformer model.
        second_waiting_toks: Positional indices for the second operand tokens.
        third_waiting_toks: Positional indices for the third operand tokens (operator).
        second_representations: CAMA representations for the second operand.
        third_representations: CAMA representations for the third operand (operator).
        second_operand_pos: The token position of the second operand.
        second_waiting_offset: Offset for accessing CAMA second operand representations.
        third_waiting_offset: Offset for accessing CAMA third operand representations.
        eval_examples: A list of (prompt, answer) tuples for evaluation.
        test_config: Dictionary containing the general configuration of waiting and peeking layers.
        batch_size: The batch size for evaluation.

    Returns:
        The accuracy of the model on the task with the given configuration.
    """
    wait_until_layer = test_config["wait_until_layer"]
    full_peek_period = test_config["full_peek_period"]

    def cama_invervention_hook(second_reps: Dict, third_reps: Dict, wait_layer: int, second_vals: List[int]):
        """Factory for a hook to patch the residual stream with CAMA representations."""
        def hook(tensor: torch.Tensor, hook: lens.hook_points.HookPoint):
            assert hook.name is not None
            layer_no = int(hook.name.split(".")[1])
            if layer_no != wait_layer:
                raise ValueError("Layer no. mismatch.")
            B, seq_len, d = tensor.shape
            out = tensor.clone()
            for i in range(B):
                sv = second_vals[i]
                # second group
                for pos in second_waiting_toks:
                    if pos >= seq_len:
                        break
                    rep = second_reps[sv][wait_layer][pos - second_waiting_offset].unsqueeze(0)
                    out[i, pos, :] = rep
                # third group
                for pos in third_waiting_toks:
                    if pos >= seq_len:
                        break
                    rep = third_reps[wait_layer][pos - third_waiting_offset].unsqueeze(0)
                    out[i, pos, :] = rep
            return out
        return hook

    def make_ABP_hook(num_allowed: int):
        """Factory for a hook to create an ABP mask."""
        def ABP_hook(tensor: torch.Tensor, hook: lens.hook_points.HookPoint):
            _, _, Q, K = tensor.shape
            q_pos = torch.arange(Q, device=tensor.device).view(Q, 1)
            k_pos = torch.arange(K, device=tensor.device).view(1, K)
            base = (q_pos >= 2) & (k_pos < q_pos) & (k_pos >= 1)
            last_q = Q - 1
            exc = (q_pos == last_q) & (k_pos >= K - num_allowed)
            mask2d = base & ~exc
            return tensor.masked_fill(mask2d.unsqueeze(0).unsqueeze(0), float('-inf'))
        return ABP_hook

    def chunks(lst: List, n: int):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i+n]

    correct = total = 0
    for batch in chunks(eval_examples, batch_size):
        prompts, answers = zip(*batch)
        token_seqs = [model.to_tokens(p, prepend_bos=True)[0] for p in prompts]
        batch_tokens = torch.stack(token_seqs, dim=0)
        
        # Extract second operand per example (for CAMA intervention)
        second_vals = []
        for seq in token_seqs:
            toks = model.to_str_tokens(seq.unsqueeze(0))
            second_vals.append(int(toks[second_operand_pos]))

        # CAMA intervention 
        cama_hook = cama_invervention_hook(
            second_representations, third_representations,
            wait_until_layer, second_vals
        )
        hooks = [(f"blocks.{wait_until_layer}.hook_resid_post", cama_hook)]

        # Information transfer period 
        full_peek_hook = make_ABP_hook(10)
        for L in range(wait_until_layer+1,
                       min(wait_until_layer+full_peek_period+1, model.cfg.n_layers)):
            hooks.append((f"blocks.{L}.attn.hook_attn_scores", full_peek_hook))
        
        # Self peek and computation period
        self_peek_hook = make_ABP_hook(1) # self peek
        for L in range(wait_until_layer+full_peek_period+1, model.cfg.n_layers):
            hooks.append((f"blocks.{L}.attn.hook_attn_scores", self_peek_hook))
        
        with model.hooks(hooks):
            logits = model(batch_tokens)
        pred_ids = logits[:, -1, :].argmax(dim=-1)
        pred_strs = [model.to_str_tokens(pid.unsqueeze(0))[
            0].strip() for pid in pred_ids]
        for pred_str, true_ans in zip(pred_strs, answers):
            try:
                p = int(pred_str)
            except:
                p = -1
            if p == int(true_ans):
                correct += 1
            total += 1

    accuracy = correct / total
    logging.info(
        f"Layer {wait_until_layer}, full-peek period {full_peek_period}: Accuracy {accuracy:.4f}")
    return accuracy

def make_prompts_and_answers(model: lens.HookedTransformer, num_prompts: int, max_op: int, max_answer_value: int, operator: str) -> Tuple[List[Tuple[str, str]], float]:
    """
    Generates prompts and answers for the arithmetic task, filtering for those the model gets correct.

    Args:
        model: The HookedTransformer model.
        num_prompts: The number of correct prompts to generate.
        max_op: The maximum value for operands.
        max_answer_value: The maximum value for the result.
        operator: The arithmetic operator to use ('+' or '-').

    Returns:
        A tuple containing:
        - A list of (prompt, answer) tuples that the model answered correctly.
        - The base accuracy of the model on the generated prompts.
    """
    prompts_and_answers = []
    expressions = []
    for op1 in tqdm(range(1, max_op)):
        for op2 in range(1, max_op):
            result = 0
            prompt = ""
            operations = {
                '+': lambda x, y: x + y,
                '-': lambda x, y: x - y,
            }
            result = operations[operator](op1, op2)
            prompt = f"{op1}{operator}{op2}="
            if prompt and 0 <= result < max_answer_value:
                expressions.append(prompt)

    if not expressions:
        raise ValueError(
            "No valid expressions were generated with the given constraints.")

    random.shuffle(expressions)

    correct_count = 0
    incorrect_count = 0
    index = 0
    pbar = tqdm(total=num_prompts, desc="Correct Prompts Found", leave=False)

    while correct_count < num_prompts and index < len(expressions):
        if index >= len(expressions):
            random.shuffle(expressions)
            index = 0

        prompt = expressions[index]
        index += 1
        tokens = model.to_tokens(prompt, prepend_bos=True)
        logits = model(tokens, return_type='logits')
        pred_id = logits[:, -1, :].argmax(dim=-1)
        predicted_token = model.to_str_tokens(
            pred_id[0].unsqueeze(0))[0].strip()

        try:
            correct_answer = eval(prompt.rstrip(" ="))
            if int(predicted_token) == correct_answer:
                correct_count += 1
                prompts_and_answers.append((prompt, predicted_token))
                pbar.update(1)
            else:
                incorrect_count += 1
        except Exception as e:
            incorrect_count += 1

    pbar.close()
    logging.info(
        f"Prompt generation accuracy: {correct_count / (correct_count + incorrect_count)}, total prompts: {correct_count + incorrect_count}")
    logging.info(
        f"for operator {operator}, total prompts: {correct_count}, acc = {correct_count / (correct_count + incorrect_count)}")

    return prompts_and_answers, (correct_count / (correct_count + incorrect_count))


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run waiting experiment on a transformer model.")
    parser.add_argument("--model_name", type=str, default="llama3-8b",
                        choices=["llama3-8b", "llama3.1-8b", "pythia", "gpt-j"], help="Model to use.")
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for computation.")
    parser.add_argument("--num_prompts", type=int, default=1000,
                        help="Number of test examples to generate.")
    parser.add_argument("--max_op", type=int, default=100,
                        help="Maximum operand value for prompt generation.")
    parser.add_argument("--max_answer_value", type=int, default=300,
                        help="Maximum answer value for prompt generation.")
    parser.add_argument("--wait_until_layer", type=int, default=14,
                        help="Layer to patch activations at.")
    parser.add_argument("--full_peek_period", type=int, default=2,
                        help="Number of layers for the 'full peek' information transfer period.")
    return parser.parse_args()


def load_model(model_name: str, device: str) -> lens.HookedTransformer:
    """Loads a HookedTransformer model."""
    logging.info(f"Loading model: {model_name} on {device}...")
    model_name_map = {
        "llama3-8b": "meta-llama/Meta-Llama-3-8B",
        "llama3.1-8b": "meta-llama/Llama-3.1-8B",
        "pythia": "EleutherAI/pythia-6.9b",
        "gpt-j": "EleutherAI/gpt-j-6b",
    }
    model_name_hf = model_name_map[model_name]

    model = lens.HookedTransformer.from_pretrained(
        model_name_hf,
        fold_ln=True,
        center_unembed=True,
        center_writing_weights=True,
        device=device
    )
    logging.info("Model loaded.")
    return model


def load_representations(load_dir: str) -> Tuple[Dict, Dict]:
    """Loads cached representations from pickle files."""
    logging.info(f"Loading representations from: {load_dir}")
    if not os.path.exists(load_dir):
        raise FileNotFoundError(f"Representation directory does not exist: {load_dir}")

    with open(os.path.join(load_dir, "second_reprs.pkl"), "rb") as f:
        second_reprs = pickle.load(f)
    logging.info("Loaded second representations.")

    with open(os.path.join(load_dir, "third_reprs.pkl"), "rb") as f:
        third_reprs = pickle.load(f)
    logging.info("Loaded third representations.")

    return second_reprs, third_reprs


def main():
    """Main function to run the experiment."""
    args = parse_args()

    # --- Configuration ---
    OPERATORS = ['+', '-']
    LOAD_DIRS = [f"{args.model_name}/plus_reprs",
                 f"{args.model_name}/minus_reprs"]

    SECOND_POSITIONS = [2, 3]  # Token positions for the second operand CAMA group
    THIRD_POSITIONS = [4]      # Token position for the final CAMA group (named third since first positions=first operand, second positions=second operand)
    SECOND_OPERAND_POS = 3     # Token position of the second operand

    # Offset to align CAMA representation index with token position
    SECOND_WAITING_OFFSET = SECOND_POSITIONS[0]
    THIRD_WAITING_OFFSET = THIRD_POSITIONS[0]

    test_config = {
        "wait_until_layer": args.wait_until_layer,
        "full_peek_period": args.full_peek_period,
    }

    logging.info("Starting...")
    model = load_model(args.model_name, args.device)

    results = {}
    for i, operator in enumerate(OPERATORS):
        current_load_dir = LOAD_DIRS[i]
        logging.info(f"--- Running experiment for operator: {operator} ---")

        second_reprs, third_reprs = load_representations(current_load_dir)

        logging.info("Generating test examples...")
        test_examples, base_acc = make_prompts_and_answers(
            model,
            num_prompts=args.num_prompts,
            max_op=args.max_op,
            max_answer_value=args.max_answer_value,
            operator=operator
        )

        logging.info("Running waiting experiment...")
        acc = perform_waiting_experiment(
            model=model,
            second_waiting_toks=SECOND_POSITIONS,
            third_waiting_toks=THIRD_POSITIONS,
            second_representations=second_reprs,
            third_representations=third_reprs,
            second_operand_pos=SECOND_OPERAND_POS,
            second_waiting_offset=SECOND_WAITING_OFFSET,
            third_waiting_offset=THIRD_WAITING_OFFSET,
            test_config=test_config,
            eval_examples=test_examples
        )
        results[operator] = {'AF1 faith': acc, 'Base Accuracy': base_acc}

    # --- Print Final Results ---
    print("\n--- Final Results ---")
    print(f"Model: {args.model_name}, Wait Layer: {args.wait_until_layer}, Peek Period: {args.full_peek_period}")
    print(f"{'Operator':<10} | {'AF1 Faith':<15} | {'Base Accuracy':<15}")
    print("-" * 45)
    for operator, metrics in results.items():
        print(
            f"{operator:<10} | {metrics['AF1 faith']:<15.4f} | {metrics['Base Accuracy']:<15.4f}")


if __name__ == "__main__":
    main()
