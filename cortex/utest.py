from colorama import Fore, Style
import cortex.cortex as ctx

def pass_fail(cond, *args):

    print(f'[ {Fore.GREEN}Passed{Style.RESET_ALL} ]' if cond else f'[ {Fore.RED}Failed{Style.RESET_ALL} ]', *args)

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
