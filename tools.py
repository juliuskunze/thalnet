import logging
import sys
from itertools import groupby
from pathlib import Path
from time import strftime

from collections import OrderedDict, Counter
from logging import getLogger, StreamHandler
from os import makedirs, path
from typing import List, Iterable, TypeVar, Callable, Optional, Dict, Tuple, Any

def timestamp() -> str:
    return strftime("%Y%m%d-%H%M%S")