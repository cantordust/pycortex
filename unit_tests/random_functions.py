import cortex.utest as utest
import cortex.random as Rand

def test_random_functions():

    print("ND:", Rand.ND())
    print("negND:", Rand.negND())
    print("posND:", Rand.posND())
    print("uni_real(-100.0,100.0):", Rand.uni_real(-100.0, 100.0))
    print("uni_int(-100,100):", Rand.uni_int(-100, 100))

    print("\n===[ Roulette wheel selection ]===\n")
    weights = [5, 3, 100, 67, 22, 0, 1e-3]
    array = [n for n in range(len(weights))]

    wheel = Rand.RouletteWheel()
    print("Array:", array)
    print("Weights:", weights)
    for index, elem in enumerate(array):
        wheel.add(elem, weights[index])

    samples = {}
    draws = 10000
    for _ in range(draws):
        sample = wheel.spin()
        if sample in samples:
             samples[sample] += 1
        else:
            samples[sample] = 1

    print("\nArray samples ({} draws):".format(draws))
    for key in sorted(samples.keys()):
        print("\t", key, ":", samples[key])

    print("\n===[ Random key and value from table ]===\n")

    table = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}

    print("Table:")
    for key in sorted(table.keys()):
        print("\t", key, ":", table[key])

    print("key:", Rand.key(table))
    print("val:", Rand.val(table))

if __name__ == '__main__':
    utest.run(test_random_functions)
