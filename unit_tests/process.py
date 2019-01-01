import torch.multiprocessing as tm

def func(_conf = None):
    print(_conf)

def main(_conf):

    assert conf is not None, "Invalid configuration"

    context = tm.get_context('forkserver')
    process = context.Process(target=func, args=(_conf,))
    process.start()
    process.join()

if __name__ == '__main__':

    func()

    conf = tm.Manager().Namespace()

    conf.x = 1000

    main(conf)
