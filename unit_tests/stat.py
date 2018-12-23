import cortex.utest as utest
import cortex.statistics as Stat

def test_stat(_type = Stat.MAType.Simple):

    stat = Stat.SMAStat("Test stats") if _type == Stat.MAType.Simple else Stat.EMAStat()
    for i in range(1, 100):
        stat.update(i)

    stat.print()

if __name__ == '__main__':
    utest.run(test_stat, _type = Stat.MAType.Simple)
    utest.run(test_stat, _type = Stat.MAType.Exponential)
