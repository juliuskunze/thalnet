import logging
import sys

from logging import getLogger, StreamHandler
from typing import List, Iterable, TypeVar, Any

E = TypeVar('Element')


def paginate(sequence: List[E], page_size: int) -> Iterable[List[E]]:
    for start in range(0, len(sequence), page_size):
        yield sequence[start:start + page_size]


logger = getLogger("results")
logger.setLevel(logging.INFO)

handler = StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
logger.addHandler(handler)


def log(obj: Any):
    logger.info(str(obj))
