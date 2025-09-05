# Old naive implementation based on this video by Andrej Karpathy: https://www.youtube.com/watch?v=zduSFxRajkE
#
# import argparse
# import os
# from collections import Counter, defaultdict
# from typing import DefaultDict, Tuple, List, Dict
#
#
# def load_training_data(train_path: str) -> str:
#     """Load and return raw text for training."""
#     training_string = ""
#     with open(train_path, "r", encoding="utf-8") as train:
#         content = train.read()
#         training_string += content
#     return training_string
#
#
# def train_bpe_tokenizer(
#     text: str, vocab_size: int
# ) -> Tuple[Dict[int, bytes], Dict[int, Tuple[int, int]]]:
#     """Learn BPE merges and return vocabulary and merge rules."""
#     if vocab_size < 256:
#         raise ValueError(
#             "Vocabulary size cannot be less than 256 for byte level BPE tokenization"
#         )
#
#     tokens = list(text.encode("utf-8"))
#     num_merges = vocab_size - 256
#     merges: Dict[int, Tuple[int, int]] = {}
#
#     for i in range(num_merges):
#         bigram_list = [(tokens[j], tokens[j + 1]) for j in range(len(tokens) - 1)]
#         if not bigram_list:
#             break
#         bigram_counter = Counter(bigram_list)
#         top_pair = max(bigram_counter, key=bigram_counter.get)
#
#         new_idx = 256 + i
#         if i % 10 == 0:
#             print(f"Merge: {i}  Top Pair: {top_pair} -> {new_idx}")
#
#         merges[new_idx] = top_pair
#
#         j = 0
#         merged_tokens = []
#         while j < len(tokens):
#             if j < len(tokens) - 1 and (tokens[j], tokens[j + 1]) == top_pair:
#                 merged_tokens.append(new_idx)
#                 j += 2
#             else:
#                 merged_tokens.append(tokens[j])
#                 j += 1
#         tokens = merged_tokens
#
#
#     vocab: Dict[int, bytes] = {}
#     for i in range(256):
#         vocab[i] = bytes([i])
#     for idx, (token1, token2) in merges.items():
#         vocab[idx] = vocab[token1] + vocab[token2]
#
#     print(vocab)
#     print(merges)
#
#     return vocab, merges
#
#
# def save_vocab(vocab: Dict[int, bytes], rollno: str, vocab_size: int) -> None:
#     """Save vocabulary file in the required format."""
#     fname = f"{rollno}_assignment2_bpe_vocab_{vocab_size}.txt"
#     special = ["<pad>", "<unk>", "<s>", "</s>"]
#     with open(fname, "w", encoding="utf-8") as f:
#         for token in special:
#             f.write(token + "\n")
#
#         sorted_vocab = sorted(vocab.items())
#         for _, token_byte in sorted_vocab:
#             try:
#                 line = token_byte.decode("utf-8")
#             except UnicodeDecodeError:
#                 line = "".join(f"<0x{byte:02x}>" for byte in token_byte)
#             f.write(line + "\n")
#
#
# def tokenize(text: str, merges: Dict[int, Tuple[int, int]]) -> List[int]:
#     """Tokenize input text using the trained BPE merge rules."""
#     tokens = list(text.encode("utf-8"))
#
#     inverted_merges: Dict[Tuple[int, int], int] = {v: k for k, v in merges.items()}
#
#     while True:
#         bigrams = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
#
#         best_pair = min(
#             (p for p in bigrams if p in inverted_merges),
#             key=lambda p: inverted_merges[p],  # Lower ID means it was learned earlier
#             default=None,
#         )
#
#         if best_pair is None:
#             break
#
#         new_token_id = inverted_merges[best_pair]
#         new_tokens = []
#         i = 0
#         while i < len(tokens):
#             if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best_pair:
#                 new_tokens.append(new_token_id)
#                 i += 2
#             else:
#                 new_tokens.append(tokens[i])
#                 i += 1
#         tokens = new_tokens
#
#     return tokens
#
#
# def detokenize(tokens: List[int], vocab: Dict[int, bytes]) -> str:
#     """Detokenize a list of token IDs back to the original text."""
#     byte_sequence = b"".join(vocab.get(token_id, b"") for token_id in tokens)
#     return byte_sequence.decode("utf-8", errors="replace")
#
#
# def save_tokens(tokens: List[int], rollno: str) -> None:
#     """Save a list of token IDs to a file, one per line."""
#     fname = f"{rollno}_assignment2_bpe_tokens.txt"
#     with open(fname, "w", encoding="utf-8") as f:
#         for tok_id in tokens:
#             f.write(str(tok_id) + "\n")
#
#
# def save_detokenized(text: str, rollno: str) -> None:
#     """Save the detokenized text to a file."""
#     fname = f"{rollno}_assignment2_bpe_detokenized.txt"
#     with open(fname, "w", encoding="utf-8") as f:
#         f.write(text)
#
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--train", type=str, required=True)
#     parser.add_argument("--input", type=str, required=True)
#     parser.add_argument("--vocab_size", type=int, required=True)
#     args = parser.parse_args()
#
#     rollno = "251110405"
#
#     # Training
#     train_text = load_training_data(args.train)
#     # train_bpe_tokenizer(train_text, args.vocab_size)
#     vocab, merges = train_bpe_tokenizer(train_text, args.vocab_size)
#     save_vocab(vocab, rollno, args.vocab_size)
#
#     # Tokenization
#     with open(args.input, "r", encoding="utf-8") as f:
#         sample_text = f.read()
#     tokens = tokenize(sample_text, merges)
#     save_tokens(tokens, rollno)
#
#     # Detokenization
#     detok_text = detokenize(tokens, vocab)
#     save_detokenized(detok_text, rollno)
#
#     # Verification
#     print("\nVerification:")
#     print(f"Original text sample: '{sample_text[:100]}...'")
#     print(f"Detokenized text sample: '{detok_text[:100]}...'")
#     assert sample_text == detok_text

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

    # Pre-allocate maximum possible size to avoid repeated reallocations
    max_tokens = len(tokens)
    working_tokens = np.zeros(max_tokens, dtype=np.int32)
    working_tokens[: len(tokens)] = tokens
    current_length = len(tokens)

    for i in range(num_merges):
        if current_length <= 1:
            break

        # Use numpy for fast bigram counting - simplified approach
        # Create arrays of consecutive pairs
        left_tokens = working_tokens[: current_length - 1]
        right_tokens = working_tokens[1:current_length]

        # Use a simple but fast approach: create compound keys and use numpy's unique
        # Multiply left by large number and add right to create unique pair IDs
        max_token_value = working_tokens[:current_length].max()
        multiplier = max_token_value + 1
        pair_ids = left_tokens * multiplier + right_tokens

        # Count using numpy's unique function
        unique_pairs, counts = np.unique(pair_ids, return_counts=True)

        if len(unique_pairs) == 0:
            break

        # Find the most frequent pair
        max_idx = np.argmax(counts)
        most_frequent_pair_id = unique_pairs[max_idx]
        max_count = counts[max_idx]

        # Extract the original pair
        left_token = int(most_frequent_pair_id // multiplier)
        right_token = int(most_frequent_pair_id % multiplier)
        top_pair = (left_token, right_token)

        new_idx = 256 + i
        if i % 50 == 0:
            print(f"Merge: {i}  Top Pair: {top_pair} -> {new_idx} (count: {max_count})")

        merges[new_idx] = top_pair

        # Fast merge operation using numpy - keep your exact logic
        # Find all positions where the pair occurs
        pair_mask = (left_tokens == left_token) & (right_tokens == right_token)
        pair_positions = np.where(pair_mask)[0]

        if len(pair_positions) == 0:
            continue

        # Vectorized merge - keep your exact approach
        new_length = current_length - len(pair_positions)
        new_tokens = np.zeros(new_length, dtype=np.int32)

        write_idx = 0
        read_idx = 0

        for pair_pos in pair_positions:
            # Copy tokens before this pair
            copy_len = pair_pos - read_idx
            if copy_len > 0:
                new_tokens[write_idx : write_idx + copy_len] = working_tokens[
                    read_idx:pair_pos
                ]
                write_idx += copy_len

            # Insert merged token
            new_tokens[write_idx] = new_idx
            write_idx += 1
            read_idx = pair_pos + 2

        # Copy remaining tokens
        remaining_len = current_length - read_idx
        if remaining_len > 0:
            new_tokens[write_idx : write_idx + remaining_len] = working_tokens[
                read_idx:current_length
            ]

        # Update working array
        working_tokens[:new_length] = new_tokens
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

    # Convert merges to sorted list for priority order
    sorted_merges = sorted(merges.items())

    max_tokens = len(tokens)
    working_tokens = np.zeros(max_tokens, dtype=np.int32)
    working_tokens[: len(tokens)] = tokens
    current_length = len(tokens)

    # Process merges in priority order (lower ID = higher priority)
    for merge_id, (left, right) in sorted_merges:
        if current_length <= 1:
            break

        # Use vectorized operations to find all merge positions at once
        left_tokens = working_tokens[: current_length - 1]
        right_tokens = working_tokens[1:current_length]

        # Find all positions where this merge can be applied
        merge_mask = (left_tokens == left) & (right_tokens == right)
        merge_positions = np.where(merge_mask)[0]

        if len(merge_positions) == 0:
            continue

        # Remove overlapping positions (greedy left-to-right)
        filtered_positions = []
        last_pos = -2
        for pos in merge_positions:
            if pos > last_pos + 1:  # No overlap with previous merge
                filtered_positions.append(pos)
                last_pos = pos

        if not filtered_positions:
            continue

        # Apply all non-overlapping merges at once
        num_merges = len(filtered_positions)
        new_length = current_length - num_merges
        new_tokens = np.zeros(new_length, dtype=np.int32)

        old_idx = 0
        new_idx = 0

        for pos in filtered_positions:
            # Copy tokens before this merge position
            copy_len = pos - old_idx
            if copy_len > 0:
                new_tokens[new_idx : new_idx + copy_len] = working_tokens[old_idx:pos]
                new_idx += copy_len

            # Add the merged token
            new_tokens[new_idx] = merge_id
            new_idx += 1
            old_idx = pos + 2

        # Copy remaining tokens after last merge
        if old_idx < current_length:
            remaining = current_length - old_idx
            new_tokens[new_idx : new_idx + remaining] = working_tokens[
                old_idx:current_length
            ]

        # Update working array
        working_tokens[:new_length] = new_tokens
        current_length = new_length

    return working_tokens[:current_length].tolist()


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
