from configs.path import logName
import sys
import logging
from functools import wraps

def log(raise_error = True):
    logging.basicConfig (
     level=logging.DEBUG, 
     format= '%(asctime)s - %(levelname)-10s - %(filename)s - %(funcName)s - %(message)s',
     handlers= [logging.FileHandler(logName, mode= 'a+')]
     )
    def decorator_factory(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            args_repr = [repr(a) for a in args]
            kwargs_repr = [f"{k}={v}" for k, v in kwargs.items()]
            params = ", ".join(args_repr + kwargs_repr)
            logging.debug(f'{function.__name__} function called with params {params}')
            try:
                result = function(*args, **kwargs)
                if isinstance(result, str):
                    logging.info(f'{function.__name__} function executed successfully with the following msg : --- {result} ---')
                return result
            except Exception as e:
                logging.error(f'{function.__name__} function rasised exception ---{sys.exc_info()[-1].tb_lineno}|;{type(e).__name__}|;{str(e)}---')
                if raise_error:
                    raise e
                return None
        return wrapper
    return decorator_factory
