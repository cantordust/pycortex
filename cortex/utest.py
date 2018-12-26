from colorama import Fore, Style
from . import cortex as ctx

def pass_fail(cond, *args):

    color = Fore.GREEN if cond else Fore.RED
    text = 'Passed' if cond else 'Failed'
    print('[ ' + color + text + Style.RESET_ALL + ' ]', *args)

    return cond

"""
Print a header with some information and run a unit test.
"""

def run(_func,
        *_args,
        **_keywords):

    ctx.UnitTestMode = True

    print("\n==================================[ Unit test ]==================================")
    print("Function:", _func.__name__)
    print("Arguments:")
    for arg in _args:
        print("\t", arg)
    print("Keyword arguments:")
    for key, val in _keywords.items():
        print("\t", key, ":", val)

    print("\nFunction output:")
    _func(*_args, **_keywords)
    print("\n===============================[ End of unit test ]==============================")
