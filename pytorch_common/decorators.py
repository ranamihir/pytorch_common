"""
=======
Useful Decorators
=======

Decorators are a useful way to save some typing and automate common tasks,
such as timing function execution or debugging dataframe sizes.

Please see this book chapter:
    <http://chimera.labs.oreilly.com/books/1230000000393/ch09.html>
on metaprogramming for more information and use cases.
"""
import logging
import inspect
import time
from functools import wraps

from .utils import human_time_interval

try:
    __IPYTHON__
except NameError:
    PRINT_FUNC = logging.info
else:
    PRINT_FUNC = print

__all__ = ["timing", "timing_with_param"]


def timing_with_param(*parameter_names):
    """
    Decorator for any function that reports how
    long it takes to run. The decorator will extract
    and print the requested function parameters.

    Example:
        from pytorch_common import timing
        @timing_with_param("name")
        def somefunc(name):
            # do something with name
            pass
    """
    def timing_decorator(func):
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
            PRINT_FUNC("Function '{}.{}{}' took {}".format(module_name, function_name,
                                                           logged_param_str, elapsed_human))
            return result
        return wrapper
    return timing_decorator

def timing(func):
    """
    Decorator for any function that reports how
    long it takes to run.
    A handy shortcut for `timing_with_param`
    when no parameter is reported.
    """
    return timing_with_param()(func)
