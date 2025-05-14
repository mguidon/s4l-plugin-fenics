from keyword import iskeyword


def is_valid_variable_name(name: str) -> bool:
    return name.isidentifier() and not iskeyword(name)


def number_with_suffix(number: int) -> str:
    suffix = {11: "th", 12: "th", 13: "th"}.get(
        number % 100, {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
    )

    return f"{number}{suffix}"
