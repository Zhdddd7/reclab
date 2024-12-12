import os
import inspect
import functools
from reclab  import _CACHE_DIR

def _create_dataset_directory(dataset_name):
    def decorator(fn):
        argspec = inspect.getfullargspec(fn)
        if not (
            argspec.args[0] == "root"
            and argspec.varargs is None
            and argspec.varkw is None
            and len(argspec.kwonlyargs) == 0
        ):
            raise ValueError("Internal Error: Given function {} did not adhere to standard signature.".format(fn))

        @functools.wraps(fn)
        def wrapper(root=_CACHE_DIR, *args, **kwargs):
            new_root = os.path.join(root, "datasets", dataset_name)
            if not os.path.exists(new_root):
                os.makedirs(new_root, exist_ok=True)
            return fn(root=new_root, *args, **kwargs)

        return wrapper

    return decorator

def _check_default_set(split, target_select, dataset_name):
    # 1. if split/ target_select is str -> tuple
    # 2. if split is not subset of target_select -> raise error
    if isinstance(split, str):
        split = (split,)
    if isinstance(target_select, str):
        target_select = (target_select,)
    if not isinstance(split, tuple):
        raise ValueError("Internal error: Expected split to be of type tuple.")
    if not set(split).issubset(set(target_select)):
        raise TypeError(
            "Given selection {} of splits is not supported for dataset {}. Please choose from {}.".format(
                split, dataset_name, target_select
            )
        )
    return split


def _wrap_datasets(datasets, split):
    # Wrap return value for _setup_datasets functions to support singular values instead
    # of tuples when split is a string.
    if isinstance(split, str):
        if len(datasets) != 1:
            raise ValueError("Internal error: Expected number of datasets is not 1.")
        return datasets[0]
    return datasets


def _wrap_split(fn, splits):
    # 1. check if the fn satisfy the specific signature
    # 2. use _check_default_set to validate `split`
    # 3. call fn multi times, every time corresponds with one ele in split
    # 4. use _wrap_datasets to adjust returns
    # 5. update new_fn signature
    argspec = inspect.getfullargspec(fn)
    if not (
        argspec.args[0] == "root"
        and argspec.args[1] == "split"
        and argspec.varargs is None
        and argspec.varkw is None
        and len(argspec.kwonlyargs) == 0
    ):
        raise ValueError("Internal Error: Given function {} did not adhere to standard signature.".format(fn))

    @functools.wraps(fn)
    def new_fn(root=_CACHE_DIR, split=splits, **kwargs):
        result = []
        for item in _check_default_set(split, splits, fn.__name__):
            result.append(fn(root, item, **kwargs))
        return _wrap_datasets(tuple(result), split)

    new_sig = inspect.signature(new_fn)
    new_sig_params = new_sig.parameters
    new_params = []
    new_params.append(new_sig_params["root"].replace(default=".data"))
    new_params.append(new_sig_params["split"].replace(default=splits))
    new_params += [entry[1] for entry in list(new_sig_params.items())[2:]]
    new_sig = new_sig.replace(parameters=tuple(new_params))
    new_fn.__signature__ = new_sig

    return new_fn

