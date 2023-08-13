import argparse


class MPDRNNConfig:
    def __init__(self):
        self.opt = None
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument("--alpha_weights_max", type=float, default="1.0", help="the upper limit of the values "
                                                                                        "of the alpha weight matrix")
        self.parser.add_argument("--alpha_weights_min", type=float, default="-1.0", help="the lower limit of the "
                                                                                         "values of the alpha weight "
                                                                                         "matrix")
        self.parser.add_argument("--dataset_name", type=str, default="usps")
        self.parser.add_argument("--method", type=str, default="BASE", help="BASE | C | EXP_ORT_C")
        self.parser.add_argument("--mu", type=float, default=0.0)
        self.parser.add_argument("--number_of_tests", type=int, default=1)
        self.parser.add_argument("--plot_diagrams", type=bool, default=False)
        self.parser.add_argument("--save_to_excel", type=bool, default=False)
        self.parser.add_argument("--seed", type=bool, default=False)
        self.parser.add_argument("--sigma", type=float, default=15.0)
        self.parser.add_argument("--slope", type=float, default=0.2)

    def parse(self):
        self.opt = self.parser.parse_args()

        return self.opt


class UtilsConfig:
    def __init__(self):
        self.opt = None
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument("--upper_limit", type=float, default="1.0", help="Uniform bias upper limit")
        self.parser.add_argument("--lower_limit", type=float, default="-1.0", help="Uniform bias lower limit")
        self.parser.add_argument("--constant", type=float, default="1.0")

    def parse(self):
        self.opt = self.parser.parse_args()

        return self.opt


class FCNNConfig:
    def __init__(self):
        self.opt = None
        self.parser = argparse.ArgumentParser()

        # FCNN parameters
        self.parser.add_argument("--dataset_name", type=str, default="connect4")
        self.parser.add_argument("--epochs", type=int, default=15)
        self.parser.add_argument("--patience", type=int, default=5)
        self.parser.add_argument("--disable_GPU", type=bool, default=True)
        self.parser.add_argument("--train_size", type=float, default=0.8)
        self.parser.add_argument("--slope", type=float, default=0.2)
        self.parser.add_argument("--learning_rate", type=float, default=1e-4)

    def parse(self):
        self.opt = self.parser.parse_args()

        return self.opt


class DatasetConfig:
    def __init__(self):
        self.opt = None
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument("--scale", type=bool, default=False)
        self.parser.add_argument("--normalize", type=bool, default=True)
        self.parser.add_argument("--type_of_normalization", type=str, default="minmax", help="zscore | minmax")

    def parse(self):
        self.opt = self.parser.parse_args()

        return self.opt
