class Fitness:

    Target = None

    def __init__(self):

        import cortex.statistics as Stat

        self.absolute = 0.0
        self.relative = 0.0

        # Running statistics about the absolute accuracy
        self.loss_stat = Stat.EMAStat()
