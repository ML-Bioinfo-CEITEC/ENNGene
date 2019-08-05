class Network:
    # parent network class for all defined architecture

    def __init__(self, dims={}, branches=[], hyperparams={}, labels=[]):
        self.dims = dims
        self.branches = branches
        self.hyperparams = hyperparams
        self.labels = labels

        # self.hyperparams = {
        #     "batch_size": self.args.batch_size,
        #     "dropout": self.args.dropout,
        #     "learn_rate": self.args.lr,
        #     "conv_num": self.args.conv_num,
        #     "dense_num": self.args.dense_num,
        #     "filter_num": self.args.filter_num,
        #     "epochs": self.args.epochs,
        #     "nodes": self.args.nodes
        # }

    def build_model(self, tune=False):
        # TODO Will there be any generic part or should this method be specific for each architecture?
        return
