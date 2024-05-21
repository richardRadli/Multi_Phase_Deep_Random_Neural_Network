import argparse


class BWELMConfig:
    def __init__(self):
        self.opt = None
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--dataset_name", type=str, default="shuttle",
                                 choices=["forest", "iris", "mnist", "satimages",  "shuttle"])
        self.parser.add_argument("--activation_function", type=str, default="leaky_ReLU",
                                 choices=["leaky_ReLU", "ReLU", "sigmoid", "identity", "sigmoid"])
        self.parser.add_argument("--inverse_activation_function", type=str, default="inverse_leaky_ReLU",
                                 choices=["inverse_leaky_ReLU", "logit", "atanh"])
        self.parser.add_argument("--init_type", type=str, default="orthogonal",
                                 choices=["uniform_0_1", "uniform_1_1", "xavier", "relu", "orthogonal"])
        self.parser.add_argument("--number_of_tests", type=int, default=1)
        self.parser.add_argument("--seed", type=bool, default=False)
        self.parser.add_argument("--slope", type=float, default=0.2)

    def parse(self):
        self.opt = self.parser.parse_args()

        return self.opt


class HELMConfig:
    def __init__(self):
        self.opt = None
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument("--dataset_name", type=str, default="mnist",
                                 choices=["connect4", "forest", "iris", "isolete", "letter", "mnist", "mnist_fashion",
                                          "musk2", "optdigits", "page_blocks", "satimages", "segment", "shuttle",
                                          "spambase", "usps"])
        self.parser.add_argument("--seed", type=bool, default=True)
        self.parser.add_argument("--penalty", type=float, default=2 ** -30)
        self.parser.add_argument("--scaling_factor", type=float, default=0.8)

    def parse(self):
        self.opt = self.parser.parse_args()

        return self.opt


class MPDRNNConfig:
    def __init__(self):
        self.opt = None
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--dataset_name", type=str, default="satimages",
                                 choices=["connect4", "forest", "iris", "isolete", "letter", "mnist", "mnist_fashion",
                                          "musk2", "optdigits", "page_blocks", "satimages", "segment", "shuttle",
                                          "spambase", "usps"])
        self.parser.add_argument("--method", type=str, default="BASE", choices=["BASE", "EXP_ORT", "EXP_ORT_C"])
        self.parser.add_argument("--activation", type=str, default="leaky_ReLU",
                                 choices=["ReLU", " sigmoid", "tanh", "identity", "leaky_ReLU"])
        self.parser.add_argument("--mu", type=float, default=0.0)
        self.parser.add_argument("--number_of_tests", type=int, default=1)
        self.parser.add_argument("--plot_diagrams", type=bool, default=False)
        self.parser.add_argument("--save_to_excel", type=bool, default=False)
        self.parser.add_argument("--seed", type=bool, default=True)
        self.parser.add_argument("--sigma_layer_2", type=float, default=10.0)
        self.parser.add_argument("--sigma_layer_3", type=float, default=0.1)
        self.parser.add_argument("--slope", type=float, default=0.2)
        self.parser.add_argument("--C", type=float, default=0.01)

    def parse(self):
        self.opt = self.parser.parse_args()

        return self.opt


class ViTELMConfig:
    def __init__(self):
        self.opt = None
        self.parser = argparse.ArgumentParser()
        self.parser.add_argument("--dataset_name", type=str, default="cifar10",
                                 choices=["cifar10", "mnist"])
        self.parser.add_argument("--activation_function", type=str, default="leaky_ReLU",
                                 choices=["leaky_ReLU", "ReLU", "sigmoid", "identity", "sigmoid"])
        self.parser.add_argument("--network_type", type=str, default="ViT", choices=["ViTELM", "ViT"])
        self.parser.add_argument("--vit_model_name", type=str, default="vitb16",
                                 choices=["vitb16", "vitb32, vitl16, vitl32"])
        self.parser.add_argument("--batch_size", type=int, default=64)
        self.parser.add_argument("--learning_rate", type=float, default=2e-5)
        self.parser.add_argument("--seed", type=bool, default=True)
        self.parser.add_argument("--train_set_size", type=float, default=0.8)
        self.parser.add_argument("--epochs", type=int, default=1)
        self.parser.add_argument("--load_weights", type=bool, default=False)

    def parse(self):
        self.opt = self.parser.parse_args()

        return self.opt
