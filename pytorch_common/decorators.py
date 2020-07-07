"""
=======
Useful Decorators
=======

Decorators are a useful way to save some typing and automate common tasks,
such as timing function execution or debugging dataframe sizes.
"""
import inspect
import logging
import time
import traceback
from functools import wraps

from .types import Callable, Optional
from .utils import human_time_interval

try:
    __IPYTHON__
except NameError:
    PRINT_FUNC = logging.info
else:
    PRINT_FUNC = print

__all__ = ["timing", "timing_with_param", "retry_if_exception", "monkey_patch_class_method"]


def timing_with_param(*parameter_names) -> Callable:
    """
    Decorator for any function that reports how
    long it takes to run. The decorator will extract
    and print the requested function parameters.

    E.g.:
        from pytorch_common import timing
        @timing_with_param("name")
        def somefunc(name):
            # Do something with `name`
            pass
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            elapsed = end - start
            elapsed_human = human_time_interval(elapsed)

            params = inspect.getcallargs(func, *args, **kwargs)
            logged_param = {param_name: params[param_name] for param_name in parameter_names}
            logged_param_str = f" {logged_param}" if logged_param else ""

            module_name, function_name = func.__module__, func.__qualname__
            PRINT_FUNC(f"Function '{module_name}.{function_name}{logged_param_str}' took {elapsed_human}")
            return result

        return wrapper

    return decorator


def timing(func: Callable) -> Callable:
    """
    Decorator for any function that
    reports how long it takes to run.
    A handy shortcut for `timing_with_param`
    when no parameter is reported.
    """
    return timing_with_param()(func)


def retry_if_exception(
    max_retries: int,
    exception_type: Exception,
    starting_error_message: Optional[str] = "",
    sleep_time: Optional[float] = None,
) -> Callable:
    """
    Decorator for retrying failed function after every `sleep_time`
    seconds if exception type and exception message (if provided)
    match `exception_type` and `starting_error_message` respectively.

    :param max_retries: Max number of retries
    :param exception_type: Exception type (e.g. RuntimeError)
    :param starting_error_message: Starting part of exception error message
    :param sleep_time: Seconds to wait before next retry
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exception_type as e:
                    error_message = str(e)
                    if error_message.startswith(starting_error_message) and attempt < max_retries:
                        traceback.print_exc()
                        if sleep_time is not None:
                            time.sleep(sleep_time)
                        PRINT_FUNC(
                            f"Attempt {attempt+1}: Retrying because {func.__name__} has @retry_if_exception decorator on it."
                        )
                        continue
                    else:
                        raise
                else:
                    raise

        return wrapper

    return decorator


def monkey_patch_class_method(cls: object, func_name: Optional[str] = None) -> Callable:
    """
    Monkey patch a particular method and override it with different
    implementation, or decorate it by adding extra logics.
    Original method is saved on `.{func_name}_orig` attribute.

    :param cls: Class being modified
    :param func_name: Name of method being overridden
                      If not given, name of decorated function is used.
    """

    def decorator(func: Callable) -> Callable:
        nonlocal func_name
        if func_name is None:
            func_name = func.__name__
        setattr(func, f"{func_name}_orig", getattr(cls, func_name, None))
        setattr(cls, func_name, func)
        return func

    return decorator
