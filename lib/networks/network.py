class Network:
    # parent network class for all defined architecture

    def __init__(self, branch_shapes=None, branches=None, hyperparams=None, labels=None, epochs=int):
        self.branch_shapes = branch_shapes or {}
        self.branches = branches or []
        self.hyperparams = hyperparams or {}
        self.labels = labels or []
        self.epochs = epochs

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

    def build_model(self):
        # TODO Will there be any generic part or should this method be specific for each architecture?
        return
