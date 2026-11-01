import os
from typing import BinaryIO
import regex as re
from multiprocessing import Pool
from collections import Counter, defaultdict
import time
import json
from tests.common import gpt2_bytes_to_unicode


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
    pair_idx = defaultdict(set)
    for ids, freq in word_freq.items():  # (32, 116, 104, 101, 138): 641902
        for a, b in zip(ids, ids[1:]):
            pair_counts[(a, b)] += freq
            pair_idx[(a, b)].add(ids)
    # print(f"pair_counts {pair_counts}")

    # for i in range(2):
    while next_id < vocab_size:
        # print(f"pair_counts {len(pair_counts)}")
        best = max(pair_counts, key=lambda p: (pair_counts[p], vocab[p[0]], vocab[p[1]]))  # freq, b'pair0', b'pair1'
        # print(f"best {best}: {vocab[best[0]]}, {vocab[best[1]]}")

        # update output
        vocab[next_id] = vocab[best[0]] + vocab[best[1]]
        merges.append((vocab[best[0]], vocab[best[1]]))

        # update merge (32, 116, 104, 101, 138) -> (32, 257, 101, 138)
        for ids in pair_idx[best]:
            # print(f"best {best} -> ids {ids}")
            freq = word_freq[ids]
            new_ids = []
            
            for a, b in zip(ids, ids[1:]):
                if (a, b) != best:
                    pair_counts[(a, b)] -= freq
                    pair_idx[(a, b)].discard(ids)
            
            byte_idx = 0
            idx_bound = len(ids)-1
            while byte_idx <= idx_bound:
                a = ids[byte_idx]
                if byte_idx+1 <= idx_bound and (a, ids[byte_idx+1]) == best:
                    new_ids.append(next_id)
                    byte_idx += 2
                else:
                    new_ids.append(a)
                    byte_idx += 1
            
            new_ids = tuple(new_ids)
            for a, b in zip(new_ids, new_ids[1:]):
                pair_counts[(a, b)] += freq
                pair_idx[(a, b)].add(new_ids)
            
            # print(f"ids -> {ids} new id {new_ids}")
            _ = word_freq.pop(ids)
            word_freq[new_ids] = freq
        
        pair_counts.pop(best)    
        _ = pair_idx.pop(best)

        next_id += 1
        # print(f"vocab {vocab}")
    
    return vocab, merges


def test_pre_token():
    special_tokens = ["<|endoftext|>"]

    count_pat = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    strip_pat = "|".join(re.escape(tok) for tok in special_tokens)  # re.escape read ?/|... as string, "|" = or

    file_path = "../data/TinyStoriesV2-GPT4-valid.txt"

    num_processes = 4

    args = (file_path, num_processes, strip_pat, count_pat)

    start_time = time.time()
    counts = pre_token_serial(args)
    print(f"serial pre-token time {time.time() - start_time}")

    start_time = time.time()
    counts = pre_token_parallel(args)
    print(f"parallel pre-token time {time.time() - start_time}")

    # print(f"pre token {counts}")


def gpt2_style_save(vocab, merges, save_pth):
    # save vocab 
    # gpt2 style printing, u can save without any extra process: save_dict = {v: k for k, v in vocab.items()}, don't use v.decode('utf-8', errors='ignore') as it will ignore > 127 when saving 
    gpt2_byte_encoder = gpt2_bytes_to_unicode()
    save_dict = {}
    for k, v in vocab.items():  
        save_k = ""
        for v_byte in v:
            save_k += gpt2_byte_encoder[v_byte]
        save_dict[save_k] = k
    save_dict = dict(sorted(save_dict.items(), key=lambda x: x[1]))
    with open(save_pth['vocab'], "w", encoding="utf-8") as f:
        json.dump(save_dict, f, ensure_ascii=False, indent=2)
        
    # save merge
    # gpt2 style printing, u can directly save as well: f.write(f"{a} {b}\n")    
    with open(save_pth['merges'], "w", encoding='utf-8') as f:
        for m in merges:
            save_m = []
            for i_m in m:
                save_i_m = ""
                for i_m_byte in i_m:
                    save_i_m += gpt2_byte_encoder[i_m_byte]
                save_m.append(save_i_m)
            f.write(f"{save_m[0]} {save_m[1]}\n") 
    

def a1_train_bpe_tinystories():
    file_path = "../data/TinyStoriesV2-GPT4-train.txt"
    vocab_size = 10000
    special_tokens = ["<|endoftext|>"]
    
    vocab, merges = train_bpe(file_path, vocab_size, special_tokens)
    
    save_pth = {
        'vocab': "assignments/a1/saves/a1-tinystories-vocab.json",
        'merges': "assignments/a1/saves/a1-tinystories-merges.txt"
    }
    
    gpt2_style_save(vocab, merges, save_pth)


def a1_train_bpe_expts_owt():
    file_path = "../data/TinyStoriesV2-GPT4-train.txt"
    vocab_size = 32000
    special_tokens = ["<|endoftext|>"]
    
    vocab, merges = train_bpe(file_path, vocab_size, special_tokens)
    
    save_pth = {
        'vocab': "assignments/a1/saves/a1-expts_owt-vocab.json",
        'merges': "assignments/a1/saves/a1-expts_owt-merges.txt"
    }
    
    gpt2_style_save(vocab, merges, save_pth)
    
    
def main():
    file_path = "../data/TinyStoriesV2-GPT4-valid.txt"
    vocab_size = 1000
    special_tokens = ["<|endoftext|>"]

    vocab, merges = train_bpe(file_path, vocab_size, special_tokens)
    
    print(f"final vocab {len(vocab)}, {vocab}")
    print(f"final merges {len(merges)}, {merges}")

if __name__ == "__main__":
    # test_pre_token()
    main()
    # a1_train_bpe_tinystories()
