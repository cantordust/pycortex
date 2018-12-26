import cortex.utest as utest
import cortex.functions as Func
import cortex.statistics as Stat
import math

def test_fmap(_function_name,
              _function,
              _val = 1.0):

    print(_function_name + "(" + str(_val) + "):", _function(_val))

if __name__ == '__main__':
    for enum, f in Func.fmap.items():
        if enum == Func.Type.Softmax:
            vals = [20000, 10.0, 7e-5, 0.0]
#            vals = [0, 0, 0]
            min_val = min(vals)
            print("Minimum:", min_val)
            vals = [v - min_val for v in vals]
            print("Vals:", vals)
            vals = [math.log1p(v) for v in vals]
            utest.run(test_fmap, enum.name, f, vals)
        else:
            utest.run(test_fmap, enum.name, f)
