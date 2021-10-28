# See https://github.com/pytorch/pytorch/pull/24895/files


from typing import List, Callable, Union, Any, TypeVar, Tuple, Iterable, Dict
from types import FunctionType

import numpy as np
import torch.tensor
import pathlib
# from torch import tensor as Tensor

T = TypeVar('T')
#Tensor = TypeVar('torch.tensor')
#Module = TypeVar('torch.nn.modules.module.Module')

Tensor = torch.Tensor

tensor_or_ndarray = Union[torch.tensor, np.ndarray]

_path_t = Union[pathlib.Path, str]
_sequence_t = Union[List[float], Tuple[float], np.ndarray]

_scalar_or_tuple_any_t = Union[T, Tuple[T, ...]]
_scalar_or_tuple_1_t = Union[T, Tuple[T]]
_scalar_or_tuple_2_t = Union[T, Tuple[T, T]]
_scalar_or_tuple_3_t = Union[T, Tuple[T, T, T]]

_size_any_t = _scalar_or_tuple_any_t[int]
_size_1_t = _scalar_or_tuple_1_t[int]
_size_2_t = _scalar_or_tuple_2_t[int]
_size_3_t = _scalar_or_tuple_3_t[int]