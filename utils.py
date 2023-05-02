def idx_to_letter(idx: int) -> str:
    return chr(ord('A') + idx)


def letter_to_idx(letter: str) -> int:
    return ord(letter) - ord('A')
