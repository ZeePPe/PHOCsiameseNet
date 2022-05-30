import argparse

class getTrainOptions():

    def __init__(self):
        self.parser = argparse.ArgumentParser(description="Parses command.")

        self.parser.add_argument("-wn", "--num_workers", type=int, help="Num of workers", default=1)
        self.parser.add_argument("-sn", "--samples_number", type=int, help="Num of samples per class", default=5)
        self.parser.add_argument("-lr", "--learning_rate", type=float, help="Starting Learning Rate", default=1e-4)
        self.parser.add_argument("-ne", "--n_epochs", type=int, help="Number of training epocs", default=10)
        self.parser.add_argument("-li", "--log_interval", type=int, help="log_interval", default=10)
        self.parser.add_argument("-ss", "--sch_step", type=int, help="Step for scheduler, num ephocs after start de LR decrease", default=5)
        self.parser.add_argument("-sg", "--sch_gamma", type=float, help="Gamma for scheduler", default=0.1)
        self.parser.add_argument("-c", "--cuda_id",
                                type=lambda str_list: [int(elem) for elem in str_list.split(',')],
                                default='-1',
                                help='The ID of the GPU to use. If not specified, training is run in CPU mode.')
        self.parser.add_argument("-fn", "--frozen", type=str, help="Base pretained PHOCnet model to use", choices=["True", "False"], default="False")
        self.parser.add_argument("-bm", "--base_model", type=str, help="Base pretained PHOCnet model to use", required=True)
        self.parser.add_argument("-sm", "--save_model", type=str, help="Name to save the trained model", required=True)
        self.parser.add_argument("-tp", "--training_path", type=str, help="Path of the training folder", required=True)
        self.parser.add_argument("-sp", "--test_path", type=str, help="Path of the test folder")

    def parse(self):
        return self.parser.parse_args()
    