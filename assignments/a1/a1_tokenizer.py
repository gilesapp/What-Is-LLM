import os
from typing import BinaryIO
import regex as re
from multiprocessing import Pool
from collections import Counter, defaultdict
import time


def find_chunk_boundaries(
        file: BinaryIO,
        desired_num_chunks: int,
        split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks
   
    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def count_chunk(args):
    file_path, start, end, strip_pat, count_pat = args
    with open(file_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    counts = Counter()
    strip_chunk = re.split(strip_pat, chunk)

    for doc in strip_chunk:
        for match in re.finditer(count_pat, doc):
            counts[match.group()] += 1
            # counts[tuple(match.group().encode("utf-8"))] += 1

    return counts


def pre_token_parallel(args, num_workers=4):
    file_path, num_processes, strip_pat, count_pat = args
    with open(file_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
    tasks = [(file_path, s, e, strip_pat, count_pat)
             for s, e in zip(boundaries, boundaries[1:])]

    total = Counter()
    with Pool(num_workers) as pool:
        for partial in pool.imap_unordered(count_chunk, tasks):
            total.update(partial)

    print(f"parallel done {file_path}")
    return total


def pre_token_serial(args):
    total = Counter()
    file_path, num_processes, strip_pat, count_pat = args
    with open(file_path, "rb") as f:
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")
        print(f"boundaries {boundaries}")
        # # The following is a serial implementation, but you can parallelize this
        # # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries, boundaries[1:]):
            # implementation 1
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")

            # Run pre-tokenization on your chunk and store the counts for each pre-token
            strip_chunk = re.split(strip_pat, chunk)

            for doc in strip_chunk:
                for match in re.finditer(count_pat, doc):  # findall return list, finditer save space for large corpus
                    total[match.group()] += 1

            # implementation 2
            # partial = count_chunk((file_path, start, end, strip_pat, count_pat))
            # total.update(partial)

    print(f"serial done {file_path} ")
    return total


def train_bpe(input_path, vocab_size, special_tokens):
    count_pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    strip_pat = "|".join(re.escape(tok) for tok in special_tokens)
    num_processes = 4
    args = (input_path, num_processes, strip_pat, count_pat)

    start_time = time.time()
    counts = pre_token_parallel(args)
    print(f"pre-token time {time.time() - start_time}")
    # print(f"counts {counts}")

    # initial 256 + special tokens
    vocab = {i: bytes([i]) for i in range(256)}
    for sp_token in special_tokens:
        vocab[len(vocab)] = sp_token.encode('utf-8')
    merges = []  # list[tuple[bytes, bytes]]

    word_freq = {
        tuple(word.encode("utf-8")): freq  # u can skip this by total[tuple(match.group().encode("utf-8"))] += 1
        for word, freq in counts.items()
    }
    # print(f"word_freq {word_freq}")

    next_id = len(vocab)
    pair_counts = Counter()
    pair_idx = defaultdict(lambda: [])
    for ids, freq in word_freq.items():  # (32, 116, 104, 101): 641902
        for a, b in zip(ids, ids[1:]):
            pair_counts[(a, b)] += freq
            pair_idx[(a, b)].append(ids)
    # print(f"pair_counts {pair_counts}")

    # for i in range(10):
    while next_id <= vocab_size:
        best = max(pair_counts, key=lambda p: (pair_counts[p], p))  # freq, b'pair0', b'pair1'
        print(f"best {best}")

        # update output
        best_bytes = (best[0], best[1])
        best_bytes_merge = vocab[best[0]] + vocab[best[1]]
        # print(f"best_bytes_merge {best_bytes_merge}")

        merges.append(best_bytes)
        vocab[next_id] = best_bytes_merge

        next_id += 1
        # update merge b' ', b't', b' t': 1000
        pair_counts.pop(best_bytes)
        for ids in pair_idx[best_bytes]:
            freq = word_freq[ids]
            new_ids = []
            pre_pair = None
            after_pair = None
            for byte_idx in range(len(ids)-1):
                a = ids[byte_idx]
                b = ids[byte_idx+1]
                # print(f"ids {ids}, a {a} b {b}")
                if (a, b) == best_bytes:  # update adjacent pair
                    pre_idx = byte_idx - 1
                    after_idx = byte_idx + 2

                    if pre_idx >= 0:
                        pre_pair = (ids[pre_idx], next_id)
                        pair_counts[(ids[pre_idx], a)] -= freq
                        pair_counts[pre_pair] += freq
                        # print(f"pre_pair {pre_pair}")
                    if after_idx < len(ids):
                        after_pair = (next_id, ids[after_idx])
                        pair_counts[(b, ids[after_idx])] -= freq
                        pair_counts[after_pair] += freq
                        # print(f"after_pair {after_pair}")
                else:
                    new_ids.append(a)

            if pre_pair:
                pair_idx[pre_pair].append(ids)
            if after_pair:
                pair_idx[after_pair].append(ids)

        _ = pair_idx.pop(best_bytes)

        print(f"vocab {vocab}")

    print(f"final vocab {len(vocab)}, {vocab}")
    print(f"final merges {len(merges)}, {merges}")


def test_pre_token():
    special_tokens = ["<|endoftext|>"]

    count_pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    strip_pat = "|".join(re.escape(tok) for tok in special_tokens)  # re.escape read ?/|... as string, "|" = or

    file_path = "../../../data/TinyStoriesV2-GPT4-valid.txt"

    num_processes = 4

    args = (file_path, num_processes, strip_pat, count_pat)

    start_time = time.time()
    counts = pre_token_serial(args)
    print(f"serial pre-token time {time.time() - start_time}")

    start_time = time.time()
    counts = pre_token_parallel(args)
    print(f"parallel pre-token time {time.time() - start_time}")

    # print(f"pre token {counts}")


def main():
    file_path = "../../../data/TinyStoriesV2-GPT4-valid.txt"
    vocab_size = 1000
    special_tokens = ["<|endoftext|>"]

    train_bpe(file_path, vocab_size, special_tokens)


if __name__ == "__main__":
    # test_pre_token()
    main()
