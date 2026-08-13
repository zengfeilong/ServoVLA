ATTENTION_SEQUENCE_MULTIPLE = 8


def round_up_to_multiple(value: int, multiple: int = ATTENTION_SEQUENCE_MULTIPLE) -> int:
    value = int(value)
    multiple = int(multiple)
    if multiple <= 0:
        raise ValueError(f"multiple must be positive, got {multiple}")
    return ((value + multiple - 1) // multiple) * multiple
