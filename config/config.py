import argparse


class MPDRNNConfig:
    def __init__(self):
        self.opt = None
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--dataset_name", type=str, default="page_blocks",
                                 choices=["connect4", "forest", "iris", "isolete", "letter", "mnist", "mnist_fashion",
                                          "musk2", "optdigits", "page_blocks", "satimages", "shuttle", "spambase",
                                          "usps"])
        self.parser.add_argument("--method", type=str, default="BASE", choices=["BASE", "EXP_ORT", "EXP_ORT_C"])
        self.parser.add_argument("--activation", type=str, default="leaky_ReLU",
                                 choices=["ReLU", " sigmoid", "tanh", "identity", "leaky_ReLU"])
        self.parser.add_argument("--mu", type=float, default=0.0)
        self.parser.add_argument("--number_of_tests", type=int, default=50)
        self.parser.add_argument("--seed", type=bool, default=True)
        self.parser.add_argument("--sigma_layer_2", type=float, default=10.0)
        self.parser.add_argument("--sigma_layer_3", type=float, default=0.1)
        self.parser.add_argument("--slope", type=float, default=0.2)
        self.parser.add_argument("--C", type=float, default=0.01)
        self.parser.add_argument("--subset_percentage", type=float, default=0.3)
        self.parser.add_argument("--pruning_method", type=str, default="max_rank")

    def parse(self):
        self.opt = self.parser.parse_args()

        return self.opt
