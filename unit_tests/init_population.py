import cortex.cortex as ctx
import cortex.utest as utest

def test_init_population():

    ctx.Species.Enabled = True
    ctx.init()

if __name__ == '__main__':
    utest.run(test_init_population)