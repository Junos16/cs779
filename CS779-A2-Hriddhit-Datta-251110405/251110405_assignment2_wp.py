# import argparse
# import os
# import numpy as np
# from typing import Tuple, List, Dict
# import timeit
#
#
# def load_training_data(train_path: str) -> str:
#     """Load and return raw text for training."""
#     with open(train_path, "r", encoding="utf-8") as train:
#         return train.read()
#
#
# def train_wordpiece_tokenizer(
#     text: str, vocab_size: int
# ) -> Tuple[Dict[int, str], Dict[int, Tuple[int, int]]]:
#     """Learn WordPiece merges and return vocabulary and merge rules."""
#     if vocab_size < 256:
#         raise ValueError(
#             "Vocabulary size cannot be less than 256 for character level WordPiece tokenization"
#         )
#
#     # Start with character-level tokens (handle unicode properly)
#     chars = list(text)
#     unique_chars = list(set(chars))
#
#     # Initialize vocabulary with unique characters
#     vocab: Dict[int, str] = {}
#     char_to_id = {}
#     for i, char in enumerate(sorted(unique_chars)):
#         vocab[i] = char
#         char_to_id[char] = i
#
#     # Convert text to token IDs
#     tokens = np.array([char_to_id[char] for char in chars], dtype=np.int32)
#     num_merges = vocab_size - len(unique_chars)
#     merges: Dict[int, Tuple[int, int]] = {}
#
#     # Pre-allocate maximum possible size to avoid repeated reallocations
#     max_tokens = len(tokens)
#     working_tokens = np.zeros(max_tokens, dtype=np.int32)
#     working_tokens[: len(tokens)] = tokens
#     current_length = len(tokens)
#
#     for i in range(num_merges):
#         if current_length <= 1:
#             break
#
#         # Create arrays of consecutive pairs
#         left_tokens = working_tokens[: current_length - 1]
#         right_tokens = working_tokens[1:current_length]
#
#         # Use compound keys for pair identification
#         max_token_value = working_tokens[:current_length].max()
#         multiplier = max_token_value + 1
#         pair_ids = left_tokens * multiplier + right_tokens
#
#         # Count using numpy's unique function
#         unique_pairs, counts = np.unique(pair_ids, return_counts=True)
#
#         if len(unique_pairs) == 0:
#             break
#
#         # WordPiece scoring: find pair with highest score (count / (freq_left * freq_right))
#         best_score = -1
#         best_pair_id = None
#         best_count = 0
#
#         # Get frequency of all tokens for scoring
#         all_token_ids, all_token_counts = np.unique(
#             working_tokens[:current_length], return_counts=True
#         )
#         token_freq = dict(zip(all_token_ids, all_token_counts))
#
#         for j, pair_id in enumerate(unique_pairs):
#             left_token = int(pair_id // multiplier)
#             right_token = int(pair_id % multiplier)
#             pair_count = counts[j]
#
#             # WordPiece score: pair_count / (left_freq * right_freq)
#             left_freq = token_freq[left_token]
#             right_freq = token_freq[right_token]
#             score = pair_count / (left_freq * right_freq)
#
#             if score > best_score:
#                 best_score = score
#                 best_pair_id = pair_id
#                 best_count = pair_count
#
#         if best_pair_id is None:
#             break
#
#         # Extract the best pair
#         left_token = int(best_pair_id // multiplier)
#         right_token = int(best_pair_id % multiplier)
#         top_pair = (left_token, right_token)
#
#         new_idx = len(unique_chars) + i
#         if i % 50 == 0:
#             print(
#                 f"Merge: {i}  Top Pair: {top_pair} -> {new_idx} (score: {best_score:.6f}, count: {best_count})"
#             )
#
#         merges[new_idx] = top_pair
#
#         # Fast merge operation using numpy
#         pair_mask = (left_tokens == left_token) & (right_tokens == right_token)
#         pair_positions = np.where(pair_mask)[0]
#
#         if len(pair_positions) == 0:
#             continue
#
#         # Vectorized merge
#         new_length = current_length - len(pair_positions)
#         new_tokens = np.zeros(new_length, dtype=np.int32)
#
#         write_idx = 0
#         read_idx = 0
#
#         for pair_pos in pair_positions:
#             # Copy tokens before this pair
#             copy_len = pair_pos - read_idx
#             if copy_len > 0:
#                 new_tokens[write_idx : write_idx + copy_len] = working_tokens[
#                     read_idx:pair_pos
#                 ]
#                 write_idx += copy_len
#
#             # Insert merged token
#             new_tokens[write_idx] = new_idx
#             write_idx += 1
#             read_idx = pair_pos + 2
#
#         # Copy remaining tokens
#         remaining_len = current_length - read_idx
#         if remaining_len > 0:
#             new_tokens[write_idx : write_idx + remaining_len] = working_tokens[
#                 read_idx:current_length
#             ]
#
#         # Update working array
#         working_tokens[:new_length] = new_tokens
#         current_length = new_length
#
#     # Build final vocabulary
#     for idx, (token1, token2) in merges.items():
#         vocab[idx] = vocab[token1] + vocab[token2]
#
#     return vocab, merges
#
#
# def save_vocab(vocab: Dict[int, str], rollno: str, vocab_size: int) -> None:
#     """Save vocabulary file in the required format."""
#     fname = f"{rollno}_assignment2_wordpiece_vocab_{vocab_size}.txt"
#     special = ["<pad>", "<unk>", "<s>", "</s>"]
#     with open(fname, "w", encoding="utf-8") as f:
#         for token in special:
#             f.write(token + "\n")
#
#         sorted_vocab = sorted(vocab.items())
#         for _, token_str in sorted_vocab:
#             f.write(token_str + "\n")
#
#
# def tokenize(
#     text: str, merges: Dict[int, Tuple[int, int]], vocab: Dict[int, str]
# ) -> List[int]:
#     """Tokenize input text using the trained WordPiece merge rules."""
#     # Build character to ID mapping from vocab
#     char_to_id = {}
#     base_chars = []
#     for token_id, token_str in vocab.items():
#         if len(token_str) == 1:  # Single character tokens
#             char_to_id[token_str] = token_id
#             base_chars.append(token_id)
#
#     # Convert text to initial character tokens
#     chars = list(text)
#     try:
#         tokens = np.array([char_to_id[char] for char in chars], dtype=np.int32)
#     except KeyError as e:
#         # Handle unknown characters by skipping them or using a fallback
#         tokens = []
#         for char in chars:
#             if char in char_to_id:
#                 tokens.append(char_to_id[char])
#         tokens = np.array(tokens, dtype=np.int32)
#
#     if len(tokens) <= 1:
#         return tokens.tolist()
#
#     # Convert merges to sorted list for priority order
#     sorted_merges = sorted(merges.items())
#
#     max_tokens = len(tokens)
#     working_tokens = np.zeros(max_tokens, dtype=np.int32)
#     working_tokens[: len(tokens)] = tokens
#     current_length = len(tokens)
#
#     # Process merges in priority order (lower ID = higher priority)
#     for merge_id, (left, right) in sorted_merges:
#         if current_length <= 1:
#             break
#
#         # Use vectorized operations to find all merge positions at once
#         left_tokens = working_tokens[: current_length - 1]
#         right_tokens = working_tokens[1:current_length]
#
#         # Find all positions where this merge can be applied
#         merge_mask = (left_tokens == left) & (right_tokens == right)
#         merge_positions = np.where(merge_mask)[0]
#
#         if len(merge_positions) == 0:
#             continue
#
#         # Remove overlapping positions (greedy left-to-right)
#         filtered_positions = []
#         last_pos = -2
#         for pos in merge_positions:
#             if pos > last_pos + 1:  # No overlap with previous merge
#                 filtered_positions.append(pos)
#                 last_pos = pos
#
#         if not filtered_positions:
#             continue
#
#         # Apply all non-overlapping merges at once
#         num_merges = len(filtered_positions)
#         new_length = current_length - num_merges
#         new_tokens = np.zeros(new_length, dtype=np.int32)
#
#         old_idx = 0
#         new_idx = 0
#
#         for pos in filtered_positions:
#             # Copy tokens before this merge position
#             copy_len = pos - old_idx
#             if copy_len > 0:
#                 new_tokens[new_idx : new_idx + copy_len] = working_tokens[old_idx:pos]
#                 new_idx += copy_len
#
#             # Add the merged token
#             new_tokens[new_idx] = merge_id
#             new_idx += 1
#             old_idx = pos + 2
#
#         # Copy remaining tokens after last merge
#         if old_idx < current_length:
#             remaining = current_length - old_idx
#             new_tokens[new_idx : new_idx + remaining] = working_tokens[
#                 old_idx:current_length
#             ]
#
#         # Update working array
#         working_tokens[:new_length] = new_tokens
#         current_length = new_length
#
#     return working_tokens[:current_length].tolist()
#
#
# def detokenize(tokens: List[int], vocab: Dict[int, str]) -> str:
#     """Detokenize a list of token IDs back to the original text."""
#     return "".join(vocab.get(token_id, "") for token_id in tokens)
#
#
# def save_tokens(tokens: List[int], rollno: str) -> None:
#     """Save a list of token IDs to a file, one per line."""
#     fname = f"{rollno}_assignment2_wordpiece_tokens.txt"
#     with open(fname, "w", encoding="utf-8") as f:
#         for tok_id in tokens:
#             f.write(str(tok_id) + "\n")
#
#
# def save_detokenized(text: str, rollno: str) -> None:
#     """Save the detokenized text to a file."""
#     fname = f"{rollno}_assignment2_wordpiece_detokenized.txt"
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
#     overall_start_time = timeit.default_timer()
#
#     # Training
#     start_time = timeit.default_timer()
#     train_text = load_training_data(args.train)
#     print(
#         f"load_training_data execution time: {timeit.default_timer() - start_time:.4f} seconds"
#     )
#
#     start_time = timeit.default_timer()
#     vocab, merges = train_wordpiece_tokenizer(train_text, args.vocab_size)
#     print(
#         f"train_wordpiece_tokenizer execution time: {timeit.default_timer() - start_time:.4f} seconds"
#     )
#
#     start_time = timeit.default_timer()
#     save_vocab(vocab, rollno, args.vocab_size)
#     print(
#         f"save_vocab execution time: {timeit.default_timer() - start_time:.4f} seconds"
#     )
#
#     # Tokenization
#     start_time = timeit.default_timer()
#     with open(args.input, "r", encoding="utf-8") as f:
#         sample_text = f.read()
#     print(
#         f"Reading input file execution time: {timeit.default_timer() - start_time:.4f} seconds"
#     )
#
#     start_time = timeit.default_timer()
#     tokens = tokenize(sample_text, merges, vocab)
#     print(f"tokenize execution time: {timeit.default_timer() - start_time:.4f} seconds")
#
#     start_time = timeit.default_timer()
#     save_tokens(tokens, rollno)
#     print(
#         f"save_tokens execution time: {timeit.default_timer() - start_time:.4f} seconds"
#     )
#
#     # Detokenization
#     start_time = timeit.default_timer()
#     detok_text = detokenize(tokens, vocab)
#     print(
#         f"detokenize execution time: {timeit.default_timer() - start_time:.4f} seconds"
#     )
#
#     start_time = timeit.default_timer()
#     save_detokenized(detok_text, rollno)
#     print(
#         f"save_detokenized execution time: {timeit.default_timer() - start_time:.4f} seconds"
#     )
#
#     # Verification
#     print("\nVerification:")
#     print(f"Original text sample: '{sample_text[:100]}...'")
#     print(f"Detokenized text sample: '{detok_text[:100]}...'")
#     assert sample_text == detok_text
#
#     print(
#         f"\nOverall script execution time: {timeit.default_timer() - overall_start_time:.4f} seconds"
#     )

# import argparse
# import os
# import numpy as np
# from typing import Tuple, List, Dict
# import timeit
#
#
# def load_training_data(train_path: str) -> str:
#     """Load and return raw text for training."""
#     with open(train_path, "r", encoding="utf-8") as train:
#         return train.read()
#
#
# def train_wordpiece_tokenizer(
#     text: str, vocab_size: int
# ) -> Tuple[Dict[int, str], Dict[int, Tuple[int, int]]]:
#     """Learn WordPiece merges and return vocabulary and merge rules."""
#     if vocab_size < 256:
#         raise ValueError(
#             "Vocabulary size cannot be less than 256 for character level WordPiece tokenization"
#         )
#
#     # Start with character-level tokens (handle unicode properly)
#     chars = list(text)
#     unique_chars = list(set(chars))
#
#     # Initialize vocabulary with unique characters
#     vocab: Dict[int, str] = {}
#     char_to_id = {}
#     for i, char in enumerate(sorted(unique_chars)):
#         vocab[i] = char
#         char_to_id[char] = i
#
#     # Convert text to token IDs
#     tokens = np.array([char_to_id[char] for char in chars], dtype=np.int32)
#     num_merges = vocab_size - len(unique_chars)
#     merges: Dict[int, Tuple[int, int]] = {}
#
#     # Pre-allocate maximum possible size to avoid repeated reallocations
#     max_tokens = len(tokens)
#     working_tokens = np.zeros(max_tokens, dtype=np.int32)
#     working_tokens[: len(tokens)] = tokens
#     current_length = len(tokens)
#
#     # Pre-compute token frequencies once and update incrementally
#     token_freq = {}
#     unique_tokens, counts = np.unique(
#         working_tokens[:current_length], return_counts=True
#     )
#     for token, count in zip(unique_tokens, counts):
#         token_freq[int(token)] = int(count)
#
#     for i in range(num_merges):
#         if current_length <= 1:
#             break
#
#         # Create arrays of consecutive pairs
#         left_tokens = working_tokens[: current_length - 1]
#         right_tokens = working_tokens[1:current_length]
#
#         # Use compound keys for pair identification
#         max_token_value = max(token_freq.keys()) if token_freq else 0
#         multiplier = max_token_value + 1
#         pair_ids = left_tokens * multiplier + right_tokens
#
#         # Count using numpy's unique function
#         unique_pairs, counts = np.unique(pair_ids, return_counts=True)
#
#         if len(unique_pairs) == 0:
#             break
#
#         # WordPiece scoring: find pair with highest score (count / (freq_left * freq_right))
#         best_score = -1
#         best_pair_id = None
#         best_count = 0
#         best_left = 0
#         best_right = 0
#
#         # Use pre-computed frequencies for scoring
#         for j, pair_id in enumerate(unique_pairs):
#             left_token = int(pair_id // multiplier)
#             right_token = int(pair_id % multiplier)
#             pair_count = counts[j]
#
#             # WordPiece score: pair_count / (left_freq * right_freq)
#             left_freq = token_freq.get(left_token, 1)
#             right_freq = token_freq.get(right_token, 1)
#             score = pair_count / (left_freq * right_freq)
#
#             if score > best_score:
#                 best_score = score
#                 best_pair_id = pair_id
#                 best_count = pair_count
#                 best_left = left_token
#                 best_right = right_token
#
#         if best_pair_id is None:
#             break
#
#         top_pair = (best_left, best_right)
#         new_idx = len(unique_chars) + i
#         if i % 50 == 0:
#             print(
#                 f"Merge: {i}  Top Pair: {top_pair} -> {new_idx} (score: {best_score:.6f}, count: {best_count})"
#             )
#
#         merges[new_idx] = top_pair
#
#         # Fast merge operation using numpy
#         pair_mask = (left_tokens == best_left) & (right_tokens == best_right)
#         pair_positions = np.where(pair_mask)[0]
#
#         if len(pair_positions) == 0:
#             continue
#
#         # Update token frequencies incrementally
#         # Decrease frequency of merged tokens
#         token_freq[best_left] = max(0, token_freq[best_left] - best_count)
#         token_freq[best_right] = max(0, token_freq[best_right] - best_count)
#         # Add frequency of new merged token
#         token_freq[new_idx] = best_count
#
#         # Vectorized merge
#         new_length = current_length - len(pair_positions)
#         new_tokens = np.zeros(new_length, dtype=np.int32)
#
#         write_idx = 0
#         read_idx = 0
#
#         for pair_pos in pair_positions:
#             # Copy tokens before this pair
#             copy_len = pair_pos - read_idx
#             if copy_len > 0:
#                 new_tokens[write_idx : write_idx + copy_len] = working_tokens[
#                     read_idx:pair_pos
#                 ]
#                 write_idx += copy_len
#
#             # Insert merged token
#             new_tokens[write_idx] = new_idx
#             write_idx += 1
#             read_idx = pair_pos + 2
#
#         # Copy remaining tokens
#         remaining_len = current_length - read_idx
#         if remaining_len > 0:
#             new_tokens[write_idx : write_idx + remaining_len] = working_tokens[
#                 read_idx:current_length
#             ]
#
#         # Update working array
#         working_tokens[:new_length] = new_tokens
#         current_length = new_length
#
#     # Build final vocabulary
#     for idx, (token1, token2) in merges.items():
#         vocab[idx] = vocab[token1] + vocab[token2]
#
#     return vocab, merges
#
#
# def save_vocab(vocab: Dict[int, str], rollno: str, vocab_size: int) -> None:
#     """Save vocabulary file in the required format."""
#     fname = f"{rollno}_assignment2_wordpiece_vocab_{vocab_size}.txt"
#     special = ["<pad>", "<unk>", "<s>", "</s>"]
#     with open(fname, "w", encoding="utf-8") as f:
#         for token in special:
#             f.write(token + "\n")
#
#         sorted_vocab = sorted(vocab.items())
#         for _, token_str in sorted_vocab:
#             f.write(token_str + "\n")
#
#
# def tokenize(
#     text: str, merges: Dict[int, Tuple[int, int]], vocab: Dict[int, str]
# ) -> List[int]:
#     """Tokenize input text using the trained WordPiece merge rules."""
#     # Build character to ID mapping from vocab
#     char_to_id = {}
#     for token_id, token_str in vocab.items():
#         if len(token_str) == 1:  # Single character tokens
#             char_to_id[token_str] = token_id
#
#     # Convert text to initial character tokens
#     chars = list(text)
#     try:
#         tokens = np.array([char_to_id[char] for char in chars], dtype=np.int32)
#     except KeyError:
#         # Handle unknown characters by skipping them
#         tokens = []
#         for char in chars:
#             if char in char_to_id:
#                 tokens.append(char_to_id[char])
#         tokens = np.array(tokens, dtype=np.int32)
#
#     if len(tokens) <= 1:
#         return tokens.tolist()
#
#     # Convert merges to arrays for vectorized processing
#     merge_ids = np.array(list(merges.keys()), dtype=np.int32)
#     merge_lefts = np.array([pair[0] for pair in merges.values()], dtype=np.int32)
#     merge_rights = np.array([pair[1] for pair in merges.values()], dtype=np.int32)
#
#     max_tokens = len(tokens)
#     working_tokens = np.zeros(max_tokens, dtype=np.int32)
#     working_tokens[: len(tokens)] = tokens
#     current_length = len(tokens)
#
#     # Process merges in priority order using vectorized operations
#     for idx in range(len(merge_ids)):
#         if current_length <= 1:
#             break
#
#         merge_id = merge_ids[idx]
#         left = merge_lefts[idx]
#         right = merge_rights[idx]
#
#         # Use vectorized operations to find all merge positions at once
#         if current_length < 2:
#             continue
#
#         left_tokens = working_tokens[: current_length - 1]
#         right_tokens = working_tokens[1:current_length]
#
#         # Find all positions where this merge can be applied
#         merge_mask = (left_tokens == left) & (right_tokens == right)
#         merge_positions = np.where(merge_mask)[0]
#
#         if len(merge_positions) == 0:
#             continue
#
#         # Remove overlapping positions (greedy left-to-right)
#         if len(merge_positions) > 1:
#             # Vectorized overlap removal
#             position_diffs = np.diff(merge_positions)
#             valid_mask = np.concatenate(([True], position_diffs > 1))
#             merge_positions = merge_positions[valid_mask]
#
#         if len(merge_positions) == 0:
#             continue
#
#         # Apply all non-overlapping merges at once
#         num_merges = len(merge_positions)
#         new_length = current_length - num_merges
#
#         if new_length <= 0:
#             continue
#
#         new_tokens = np.zeros(new_length, dtype=np.int32)
#
#         # Vectorized merge application
#         old_idx = 0
#         new_idx = 0
#
#         for pos in merge_positions:
#             # Copy tokens before this merge position
#             copy_len = pos - old_idx
#             if copy_len > 0:
#                 new_tokens[new_idx : new_idx + copy_len] = working_tokens[old_idx:pos]
#                 new_idx += copy_len
#
#             # Add the merged token
#             new_tokens[new_idx] = merge_id
#             new_idx += 1
#             old_idx = pos + 2
#
#         # Copy remaining tokens after last merge
#         if old_idx < current_length:
#             remaining = current_length - old_idx
#             if remaining > 0:
#                 new_tokens[new_idx : new_idx + remaining] = working_tokens[
#                     old_idx:current_length
#                 ]
#
#         # Update working array
#         working_tokens[:new_length] = new_tokens
#         current_length = new_length
#
#     return working_tokens[:current_length].tolist()
#
#
# def detokenize(tokens: List[int], vocab: Dict[int, str]) -> str:
#     """Detokenize a list of token IDs back to the original text."""
#     return "".join(vocab.get(token_id, "") for token_id in tokens)
#
#
# def save_tokens(tokens: List[int], rollno: str) -> None:
#     """Save a list of token IDs to a file, one per line."""
#     fname = f"{rollno}_assignment2_wordpiece_tokens.txt"
#     with open(fname, "w", encoding="utf-8") as f:
#         for tok_id in tokens:
#             f.write(str(tok_id) + "\n")
#
#
# def save_detokenized(text: str, rollno: str) -> None:
#     """Save the detokenized text to a file."""
#     fname = f"{rollno}_assignment2_wordpiece_detokenized.txt"
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
#     overall_start_time = timeit.default_timer()
#
#     # Training
#     start_time = timeit.default_timer()
#     train_text = load_training_data(args.train)
#     print(
#         f"load_training_data execution time: {timeit.default_timer() - start_time:.4f} seconds"
#     )
#
#     start_time = timeit.default_timer()
#     vocab, merges = train_wordpiece_tokenizer(train_text, args.vocab_size)
#     print(
#         f"train_wordpiece_tokenizer execution time: {timeit.default_timer() - start_time:.4f} seconds"
#     )
#
#     start_time = timeit.default_timer()
#     save_vocab(vocab, rollno, args.vocab_size)
#     print(
#         f"save_vocab execution time: {timeit.default_timer() - start_time:.4f} seconds"
#     )
#
#     # Tokenization
#     start_time = timeit.default_timer()
#     with open(args.input, "r", encoding="utf-8") as f:
#         sample_text = f.read()
#     print(
#         f"Reading input file execution time: {timeit.default_timer() - start_time:.4f} seconds"
#     )
#
#     start_time = timeit.default_timer()
#     tokens = tokenize(sample_text, merges, vocab)
#     print(f"tokenize execution time: {timeit.default_timer() - start_time:.4f} seconds")
#
#     start_time = timeit.default_timer()
#     save_tokens(tokens, rollno)
#     print(
#         f"save_tokens execution time: {timeit.default_timer() - start_time:.4f} seconds"
#     )
#
#     # Detokenization
#     start_time = timeit.default_timer()
#     detok_text = detokenize(tokens, vocab)
#     print(
#         f"detokenize execution time: {timeit.default_timer() - start_time:.4f} seconds"
#     )
#
#     start_time = timeit.default_timer()
#     save_detokenized(detok_text, rollno)
#     print(
#         f"save_detokenized execution time: {timeit.default_timer() - start_time:.4f} seconds"
#     )
#
#     # Verification
#     print("\nVerification:")
#     print(f"Original text sample: '{sample_text[:100]}...'")
#     print(f"Detokenized text sample: '{detok_text[:100]}...'")
#     assert sample_text == detok_text
#
#     print(
#         f"\nOverall script execution time: {timeit.default_timer() - overall_start_time:.4f} seconds"
#     )

import argparse
import os
import numpy as np
from typing import Tuple, List, Dict
import timeit


def load_training_data(train_path: str) -> str:
    """Load and return raw text for training."""
    with open(train_path, "r", encoding="utf-8") as train:
        return train.read()


def train_wordpiece_tokenizer(
    text: str, vocab_size: int
) -> Tuple[Dict[int, str], Dict[int, Tuple[int, int]]]:
    """Learn WordPiece merges and return vocabulary and merge rules."""
    if vocab_size < 256:
        raise ValueError(
            "Vocabulary size cannot be less than 256 for character level WordPiece tokenization"
        )

    # Start with character-level tokens (handle unicode properly)
    chars = list(text)
    unique_chars = list(set(chars))

    # Initialize vocabulary with unique characters
    vocab: Dict[int, str] = {}
    char_to_id = {}
    for i, char in enumerate(sorted(unique_chars)):
        vocab[i] = char
        char_to_id[char] = i

    # Convert text to token IDs
    tokens = np.array([char_to_id[char] for char in chars], dtype=np.int32)
    num_merges = vocab_size - len(unique_chars)
    merges: Dict[int, Tuple[int, int]] = {}

    # Pre-allocate maximum possible size to avoid repeated reallocations
    max_tokens = len(tokens)
    working_tokens = np.zeros(max_tokens, dtype=np.int32)
    working_tokens[: len(tokens)] = tokens
    current_length = len(tokens)

    # Pre-compute all token frequencies once and maintain as numpy array
    max_possible_token = vocab_size
    token_freq_array = np.zeros(max_possible_token, dtype=np.int32)
    unique_tokens, counts = np.unique(
        working_tokens[:current_length], return_counts=True
    )
    token_freq_array[unique_tokens] = counts

    for i in range(num_merges):
        if current_length <= 1:
            break

        # Create arrays of consecutive pairs
        left_tokens = working_tokens[: current_length - 1]
        right_tokens = working_tokens[1:current_length]

        # Use compound keys for pair identification - more efficient multiplier
        max_token_value = len(unique_chars) + i
        multiplier = max_token_value + 1
        pair_ids = left_tokens.astype(np.int64) * multiplier + right_tokens.astype(
            np.int64
        )

        # Count using numpy's unique function
        unique_pairs, counts = np.unique(pair_ids, return_counts=True)

        if len(unique_pairs) == 0:
            break

        # Vectorized WordPiece scoring
        left_tokens_unique = (unique_pairs // multiplier).astype(np.int32)
        right_tokens_unique = (unique_pairs % multiplier).astype(np.int32)

        # Get frequencies using array indexing (much faster than dict lookup)
        left_freqs = token_freq_array[left_tokens_unique]
        right_freqs = token_freq_array[right_tokens_unique]

        # Vectorized score calculation with safety check for division by zero
        denominators = left_freqs.astype(np.float64) * right_freqs.astype(np.float64)
        scores = np.divide(
            counts.astype(np.float64),
            denominators,
            out=np.zeros_like(counts, dtype=np.float64),
            where=denominators != 0,
        )

        # Find best score
        best_idx = np.argmax(scores)
        best_score = scores[best_idx]
        best_pair_id = unique_pairs[best_idx]
        best_count = counts[best_idx]

        best_left = int(best_pair_id // multiplier)
        best_right = int(best_pair_id % multiplier)

        top_pair = (best_left, best_right)
        new_idx = len(unique_chars) + i
        if i % 50 == 0:
            print(
                f"Merge: {i}  Top Pair: {top_pair} -> {new_idx} (score: {best_score:.6f}, count: {best_count})"
            )

        merges[new_idx] = top_pair

        # Fast merge operation using numpy
        pair_mask = (left_tokens == best_left) & (right_tokens == best_right)
        pair_positions = np.where(pair_mask)[0]

        if len(pair_positions) == 0:
            continue

        # Update token frequencies using array operations
        token_freq_array[best_left] = max(0, token_freq_array[best_left] - best_count)
        token_freq_array[best_right] = max(0, token_freq_array[best_right] - best_count)
        token_freq_array[new_idx] = best_count

        # Vectorized merge - optimized for large arrays
        new_length = current_length - len(pair_positions)

        # Create boolean mask for tokens to keep
        keep_mask = np.ones(current_length, dtype=bool)
        keep_mask[pair_positions] = False
        keep_mask[pair_positions + 1] = False

        # Extract tokens to keep
        tokens_to_keep = working_tokens[:current_length][keep_mask]

        # Create new array with merged tokens inserted
        new_tokens = np.zeros(new_length, dtype=np.int32)

        # Fill in the kept tokens and merged tokens efficiently
        old_idx = 0
        new_idx_pos = 0

        for pair_pos in pair_positions:
            # Copy tokens before this pair
            copy_len = pair_pos - old_idx
            if copy_len > 0:
                new_tokens[new_idx_pos : new_idx_pos + copy_len] = working_tokens[
                    old_idx:pair_pos
                ]
                new_idx_pos += copy_len

            # Insert merged token
            new_tokens[new_idx_pos] = new_idx
            new_idx_pos += 1
            old_idx = pair_pos + 2

        # Copy remaining tokens
        remaining_len = current_length - old_idx
        if remaining_len > 0:
            new_tokens[new_idx_pos : new_idx_pos + remaining_len] = working_tokens[
                old_idx:current_length
            ]

        # Update working array
        working_tokens[:new_length] = new_tokens
        current_length = new_length

    # Build final vocabulary
    for idx, (token1, token2) in merges.items():
        vocab[idx] = vocab[token1] + vocab[token2]

    return vocab, merges


def save_vocab(vocab: Dict[int, str], rollno: str, vocab_size: int) -> None:
    """Save vocabulary file in the required format."""
    fname = f"{rollno}_assignment2_wordpiece_vocab_{vocab_size}.txt"
    special = ["<pad>", "<unk>", "<s>", "</s>"]
    with open(fname, "w", encoding="utf-8") as f:
        for token in special:
            f.write(token + "\n")

        sorted_vocab = sorted(vocab.items())
        for _, token_str in sorted_vocab:
            f.write(token_str + "\n")


def tokenize(
    text: str, merges: Dict[int, Tuple[int, int]], vocab: Dict[int, str]
) -> List[int]:
    """Tokenize input text using the trained WordPiece merge rules."""
    # Build character to ID mapping from vocab
    char_to_id = {}
    for token_id, token_str in vocab.items():
        if len(token_str) == 1:  # Single character tokens
            char_to_id[token_str] = token_id

    # Convert text to initial character tokens
    chars = list(text)
    try:
        tokens = np.array([char_to_id[char] for char in chars], dtype=np.int32)
    except KeyError:
        # Handle unknown characters by skipping them
        tokens = []
        for char in chars:
            if char in char_to_id:
                tokens.append(char_to_id[char])
        tokens = np.array(tokens, dtype=np.int32)

    if len(tokens) <= 1:
        return tokens.tolist()

    # Convert merges to sorted list and create lookup arrays
    sorted_merge_items = sorted(merges.items())
    merge_priorities = {
        merge_id: idx for idx, (merge_id, _) in enumerate(sorted_merge_items)
    }

    max_tokens = len(tokens)
    working_tokens = np.zeros(max_tokens, dtype=np.int32)
    working_tokens[: len(tokens)] = tokens
    current_length = len(tokens)

    # Keep applying merges until no more can be applied
    changed = True
    while changed and current_length > 1:
        changed = False

        # Find all possible merges in current sequence
        left_tokens = working_tokens[: current_length - 1]
        right_tokens = working_tokens[1:current_length]

        # Create pairs and find which ones have merge rules
        best_merge_id = None
        best_positions = None
        best_priority = len(sorted_merge_items)  # Lower is better

        # Check each merge rule for applicability
        for merge_id, (left, right) in sorted_merge_items:
            merge_mask = (left_tokens == left) & (right_tokens == right)
            positions = np.where(merge_mask)[0]

            if len(positions) > 0:
                priority = merge_priorities[merge_id]
                if priority < best_priority:
                    best_priority = priority
                    best_merge_id = merge_id
                    best_positions = positions
                    break  # Found highest priority merge, stop searching

        if best_merge_id is None:
            break

        # Remove overlapping positions (greedy left-to-right)
        if len(best_positions) > 1:
            filtered_positions = [best_positions[0]]
            for pos in best_positions[1:]:
                if pos > filtered_positions[-1] + 1:
                    filtered_positions.append(pos)
            best_positions = np.array(filtered_positions)

        # Apply the best merge
        num_merges = len(best_positions)
        new_length = current_length - num_merges
        new_tokens = np.zeros(new_length, dtype=np.int32)

        old_idx = 0
        new_idx = 0

        for pos in best_positions:
            # Copy tokens before this merge position
            copy_len = pos - old_idx
            if copy_len > 0:
                new_tokens[new_idx : new_idx + copy_len] = working_tokens[old_idx:pos]
                new_idx += copy_len

            # Add the merged token
            new_tokens[new_idx] = best_merge_id
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
        changed = True

    return working_tokens[:current_length].tolist()


def detokenize(tokens: List[int], vocab: Dict[int, str]) -> str:
    """Detokenize a list of token IDs back to the original text."""
    return "".join(vocab.get(token_id, "") for token_id in tokens)


def save_tokens(tokens: List[int], rollno: str) -> None:
    """Save a list of token IDs to a file, one per line."""
    fname = f"{rollno}_assignment2_wordpiece_tokens.txt"
    with open(fname, "w", encoding="utf-8") as f:
        for tok_id in tokens:
            f.write(str(tok_id) + "\n")


def save_detokenized(text: str, rollno: str) -> None:
    """Save the detokenized text to a file."""
    fname = f"{rollno}_assignment2_wordpiece_detokenized.txt"
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
    vocab, merges = train_wordpiece_tokenizer(train_text, args.vocab_size)
    print(
        f"train_wordpiece_tokenizer execution time: {timeit.default_timer() - start_time:.4f} seconds"
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
    tokens = tokenize(sample_text, merges, vocab)
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
