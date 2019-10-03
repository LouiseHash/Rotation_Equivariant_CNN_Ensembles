import argparse
from RotEqCnn import RotEqCNN


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', type=str, default='mnist')
parser.add_argument('--ensemble_num', type=int, default=9, help='Number of ensemble member')
parser.add_argument('--train_epoch', type=int, default=6, help='Number of training epochs')

opt = parser.parse_args()
dataset = opt.dataset
e_num = int(opt.ensemble_num)
t_epoch = opt.train_epoch

recnn = RotEqCNN(e_num, dataset, t_epoch)
recnn.get_dataset()
recnn.init_models()
recnn.train()
recnn.show_test_result()
