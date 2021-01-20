

class DPCounter:
    def __init__(self, args, model_args):
        self.T = 0
        self.eps = 0
        self.delta = model_args.delta
        self.should_stop = False
        self.sigma = model_args.noise_sigma
        self.q = float(args.batch_size) / (args.num_samples)