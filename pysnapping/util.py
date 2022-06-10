import typing
import itertools


def iter_consecutive_groups(
    integers: typing.Iterable[int],
) -> typing.Iterator[typing.List[int]]:
    """Iterate over groups of consecutive integers."""
    return (
        [item[1] for item in group]
        for _, group in itertools.groupby(
            enumerate(integers),
            lambda t: t[0] - t[1],
        )
    )
