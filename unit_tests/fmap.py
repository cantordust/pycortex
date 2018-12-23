import cortex.utest as utest
import cortex.functions as Func

def test_fmap(_val = 1.0):

    for enum, f in Func.fmap.items():
        print(enum.name, f(_val))

if __name__ == '__main__':
    utest.run(test_fmap)