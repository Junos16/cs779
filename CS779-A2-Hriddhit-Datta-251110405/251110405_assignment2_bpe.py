import argparse
import os
import numpy as np
from typing import Tuple, List, Dict
import timeit


def load_training_data(train_path: str) -> str:
    """Load and return raw text for training."""
    with open(train_path, "r", encoding="utf-8") as train:
        return train.read()


def train_bpe_tokenizer(
    text: str, vocab_size: int
) -> Tuple[Dict[int, bytes], Dict[int, Tuple[int, int]]]:
    """Learn BPE merges and return vocabulary and merge rules."""
    if vocab_size < 256:
        raise ValueError(
            "Vocabulary size cannot be less than 256 for byte level BPE tokenization"
        )

    # Convert to numpy array for faster operations
    tokens = np.array(list(text.encode("utf-8")), dtype=np.int32)
    num_merges = vocab_size - 256
    merges: Dict[int, Tuple[int, int]] = {}

    # Use the original tokens array directly, reallocating only when needed
    working_tokens = tokens.copy()
    current_length = len(tokens)

    for i in range(num_merges):
        if current_length <= 1:
            break

        # Optimized bigram counting using direct indexing
        if current_length == 1:
            break

        # Create views of consecutive tokens for vectorized operations
        left_view = working_tokens[: current_length - 1]
        right_view = working_tokens[1:current_length]

        # Use numpy's lexsort for efficient pair counting
        # Create structured array for pairs
        pairs_dtype = np.dtype([("left", np.int32), ("right", np.int32)])
        pairs = np.empty(current_length - 1, dtype=pairs_dtype)
        pairs["left"] = left_view
        pairs["right"] = right_view

        # Sort and count unique pairs efficiently
        sorted_idx = np.lexsort((pairs["right"], pairs["left"]))
        sorted_pairs = pairs[sorted_idx]

        # Find unique pairs and their counts using diff
        if len(sorted_pairs) == 0:
            break

        # Identify boundaries between different pairs
        pair_changes = np.concatenate(
            (
                [True],
                (sorted_pairs["left"][1:] != sorted_pairs["left"][:-1])
                | (sorted_pairs["right"][1:] != sorted_pairs["right"][:-1]),
                [True],
            )
        )

        change_indices = np.where(pair_changes)[0]
        counts = np.diff(change_indices)
        unique_pairs = sorted_pairs[change_indices[:-1]]

        if len(counts) == 0:
            break

        # Find the most frequent pair
        max_idx = np.argmax(counts)
        top_pair = (
            int(unique_pairs[max_idx]["left"]),
            int(unique_pairs[max_idx]["right"]),
        )
        max_count = counts[max_idx]

        new_idx = 256 + i
        if i % 50 == 0:
            print(f"Merge: {i}  Top Pair: {top_pair} -> {new_idx} (count: {max_count})")

        merges[new_idx] = top_pair

        # Optimized merge operation using boolean indexing
        left_token, right_token = top_pair

        # Find merge positions more efficiently
        merge_mask = (left_view == left_token) & (right_view == right_token)
        merge_positions = np.where(merge_mask)[0]

        if len(merge_positions) == 0:
            continue

        # Calculate new length and create result array
        new_length = current_length - len(merge_positions)

        # Only reallocate if we need more space than current array provides
        if new_length > len(working_tokens):
            working_tokens = np.zeros(new_length, dtype=np.int32)

        # Use numpy operations for efficient copying
        result = np.zeros(new_length, dtype=np.int32)

        # Create a mask for positions to skip (merge positions + 1)
        skip_mask = np.zeros(current_length, dtype=bool)
        skip_mask[merge_positions + 1] = True  # Skip the second token of each pair

        # Copy non-skipped tokens
        kept_positions = np.where(~skip_mask)[0]
        result[: len(kept_positions)] = working_tokens[kept_positions]

        # Replace merged pairs with new token
        merge_result_positions = merge_positions - np.arange(len(merge_positions))
        result[merge_result_positions] = new_idx

        working_tokens[:new_length] = result
        current_length = new_length

    # Build vocabulary
    vocab: Dict[int, bytes] = {}
    for i in range(256):
        vocab[i] = bytes([i])
    for idx, (token1, token2) in merges.items():
        vocab[idx] = vocab[token1] + vocab[token2]

    return vocab, merges


def save_vocab(vocab: Dict[int, bytes], rollno: str, vocab_size: int) -> None:
    """Save vocabulary file in the required format."""
    fname = f"{rollno}_assignment2_bpe_vocab_{vocab_size}.txt"
    special = ["<pad>", "<unk>", "<s>", "</s>"]
    with open(fname, "w", encoding="utf-8") as f:
        for token in special:
            f.write(token + "\n")

        sorted_vocab = sorted(vocab.items())
        for _, token_byte in sorted_vocab:
            try:
                line = token_byte.decode("utf-8")
            except UnicodeDecodeError:
                line = "".join(f"<0x{byte:02x}>" for byte in token_byte)
            f.write(line + "\n")


def tokenize(text: str, merges: Dict[int, Tuple[int, int]]) -> List[int]:
    """Tokenize input text using the trained BPE merge rules."""
    tokens = np.array(list(text.encode("utf-8")), dtype=np.int32)

    if len(tokens) <= 1:
        return tokens.tolist()

    # Pre-compute merge lookup for faster access
    # Create arrays for vectorized operations
    merge_ids = np.array(list(merges.keys()), dtype=np.int32)
    merge_lefts = np.array([pair[0] for pair in merges.values()], dtype=np.int32)
    merge_rights = np.array([pair[1] for pair in merges.values()], dtype=np.int32)

    # Sort by merge ID for correct application order
    sort_idx = np.argsort(merge_ids)
    merge_ids = merge_ids[sort_idx]
    merge_lefts = merge_lefts[sort_idx]
    merge_rights = merge_rights[sort_idx]

    current_tokens = tokens.copy()

    # Apply merges iteratively
    for i in range(len(merge_ids)):
        if len(current_tokens) <= 1:
            break

        merge_id = merge_ids[i]
        left_token = merge_lefts[i]
        right_token = merge_rights[i]

        # Vectorized search for merge opportunities
        if len(current_tokens) == 1:
            break

        left_view = current_tokens[:-1]
        right_view = current_tokens[1:]

        # Find all positions where this merge can be applied
        merge_mask = (left_view == left_token) & (right_view == right_token)
        merge_positions = np.where(merge_mask)[0]

        if len(merge_positions) == 0:
            continue

        # Apply merge using efficient array operations
        new_length = len(current_tokens) - len(merge_positions)
        new_tokens = np.zeros(new_length, dtype=np.int32)

        # Create skip mask
        skip_mask = np.zeros(len(current_tokens), dtype=bool)
        skip_mask[merge_positions + 1] = True

        # Copy tokens that aren't being skipped
        kept_indices = np.where(~skip_mask)[0]
        new_tokens[: len(kept_indices)] = current_tokens[kept_indices]

        # Replace merge positions with new token
        result_positions = merge_positions - np.arange(len(merge_positions))
        new_tokens[result_positions] = merge_id

        current_tokens = new_tokens

    return current_tokens.tolist()


def detokenize(tokens: List[int], vocab: Dict[int, bytes]) -> str:
    """Detokenize a list of token IDs back to the original text."""
    byte_sequence = b"".join(vocab.get(token_id, b"") for token_id in tokens)
    return byte_sequence.decode("utf-8", errors="replace")


def save_tokens(tokens: List[int], rollno: str) -> None:
    """Save a list of token IDs to a file, one per line."""
    fname = f"{rollno}_assignment2_bpe_tokens.txt"
    with open(fname, "w", encoding="utf-8") as f:
        for tok_id in tokens:
            f.write(str(tok_id) + "\n")


def save_detokenized(text: str, rollno: str) -> None:
    """Save the detokenized text to a file."""
    fname = f"{rollno}_assignment2_bpe_detokenized.txt"
    with open(fname, "w", encoding="utf-8") as f:
        f.write(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--vocab_size", type=int, required=True)
    args = parser.parse_args()

    rollno = "251110405"

    overall_start_time = timeit.default_timer()

    # Training
    start_time = timeit.default_timer()
    train_text = load_training_data(args.train)
    print(
        f"load_training_data execution time: {timeit.default_timer() - start_time:.4f} seconds"
    )

    start_time = timeit.default_timer()
    vocab, merges = train_bpe_tokenizer(train_text, args.vocab_size)
    print(
        f"train_bpe_tokenizer execution time: {timeit.default_timer() - start_time:.4f} seconds"
    )

    start_time = timeit.default_timer()
    save_vocab(vocab, rollno, args.vocab_size)
    print(
        f"save_vocab execution time: {timeit.default_timer() - start_time:.4f} seconds"
    )

    # Tokenization
    start_time = timeit.default_timer()
    with open(args.input, "r", encoding="utf-8") as f:
        sample_text = f.read()
    print(
        f"Reading input file execution time: {timeit.default_timer() - start_time:.4f} seconds"
    )

    start_time = timeit.default_timer()
    tokens = tokenize(sample_text, merges)
    print(f"tokenize execution time: {timeit.default_timer() - start_time:.4f} seconds")

    start_time = timeit.default_timer()
    save_tokens(tokens, rollno)
    print(
        f"save_tokens execution time: {timeit.default_timer() - start_time:.4f} seconds"
    )

    # Detokenization
    start_time = timeit.default_timer()
    detok_text = detokenize(tokens, vocab)
    print(
        f"detokenize execution time: {timeit.default_timer() - start_time:.4f} seconds"
    )

    start_time = timeit.default_timer()
    save_detokenized(detok_text, rollno)
    print(
        f"save_detokenized execution time: {timeit.default_timer() - start_time:.4f} seconds"
    )

    # Verification
    print("\nVerification:")
    print(f"Original text sample: '{sample_text[:100]}...'")
    print(f"Detokenized text sample: '{detok_text[:100]}...'")
    assert sample_text == detok_text

    print(
        f"\nOverall script execution time: {timeit.default_timer() - overall_start_time:.4f} seconds"
    )
