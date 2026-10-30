def main():
    # unicode standard
    char_sample = '0'
    uni_code = ord(char_sample)
    print(f"----[unicode standard]---- char_sample {char_sample} -> code point {uni_code}, char {chr(uni_code)} string")
    print(f"----[unicode standard]---- special case: chr(0) -> {chr(0)}")

    # unicode encoding (standard unicode -> large vocab size (+150k), sparse (some char rarely used)
    encode_str = "hello 世界"
    utf8_encoded = encode_str.encode('utf-8')
    print(f"----[unicode encoding]---- {encode_str} -> utf8 {utf8_encoded}")
    print(f"----[unicode encoding]---- {encode_str} -> utf8 byte value {list(utf8_encoded)}")

    # why use utf8
    utf32_encoded = encode_str.encode('utf-32')
    print(f"----[unicode encoding]---- {encode_str} -> utf32 {utf32_encoded}")
    print(f"----[unicode encoding]---- {encode_str} -> utf32 byte value {list(utf32_encoded)}")

    # wrong decode implementation -> 1 byte does not necessarily correspond to 1 Unicode char
    def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
        return "".join([bytes([b]).decode("utf-8") for b in bytestring])
    print(f"----[unicode encoding]---- wrong decod implementation {decode_utf8_bytes_to_str_wrong('hello'.encode('utf-8'))}")
    print(f"----[unicode encoding]---- wrong decode implementation decode_utf8_bytes_to_str_wrong('世界'.encode('utf-8'))")

    # correct decode -> extra note - utf8 byte decode by following a native 8-bit rule e.g. 111* **** represents a 3-byte-long head
    encode_str = "世"
    byte_seq = tuple(encode_str.encode('utf-8'))
    print(f"----[unicode encoding]---- byte sequence for {encode_str} -> {byte_seq}")
    print(f"----[unicode encoding]---- 2-byte sequence does not decode to any Unicode char: bytes([184, 150]).decode('utf-8')")
    print(f"----[unicode encoding]---- correct decode: {bytes(byte_seq).decode('utf-8')} or {bytes([228, 184, 150]).decode('utf-8')}")

    # pre tokenization
    # cons1: computationally expensive (we have to take a full pass over the corpus each time we merge)
    # cons2: directly merging bytes across the corpus may result in tokens that differ only in punctuation (hi! vs hi.)

    # Most modern tokenizers use a regex-based pre-tokenize
    import regex as re
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pre_tokens = re.findall(PAT, "some text that i'll pre-tokenize")
    pre_tokens_iter = re.finditer(PAT, "some text that i'll pre-tokenize")
    print(f"----[pre tokenization]---- pre_tokens {pre_tokens}")
    print(f"----[pre tokenization]---- pre_tokens {pre_tokens_iter}")

    # break ties -> preferring the lexicographically greater pair
    break_ties_pair_selected = max([("A", "B"), ("A", "C"), ("B", "ZZ"), ("BA", "A")])
    print(f"----[pre tokenization]---- break ties {break_ties_pair_selected}")

    break_ties_pair_selected = max([(b"A", b"B"), (b"A", b"C"), (b"B", b"ZZ"), (b"BA", b"A")])
    print(f"----[pre tokenization]---- break ties {break_ties_pair_selected}")


if __name__ == "__main__":
    main()
