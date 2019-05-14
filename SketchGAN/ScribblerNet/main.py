import argparse
from utils import *
import tensorflow as tf
import ScribblerNet

def parseArgs():
    desc = "Hair 2D orientation field recovered from mask and sketch."
    parser = argparse.ArgumentParser(description=desc)

    parser.add_argument("--train_dir", type=str,
                        default="../TrainData/",
                        help="dir to training data.")
    parser.add_argument("--test_dir", type=str,
                        default="../TestData/",
                        help="dir to testing data.")
    parser.add_argument("--if_train", type=str,
                        default=False,
                        help="True: process training. False: process testing.")
    parser.add_argument("--batch_size", type=int,
                        default=4,
                        help="Batch size set for training process.")
    parser.add_argument("--rst_dir", type=str,
                        default="./RstData/",
                        help="dir to save result.")
    parser.add_argument("--input_dir", type=str,
                        default="../../TestData/0/",
                        help="dir of Input data not for training.")

    args = parser.parse_args()

    return args

def main():
    args = parseArgs()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True
    with tf.Session(config=run_config) as sess:
        train_model = ScribblerNet.Model(sess, train_root=args.train_dir, test_root=args.test_dir,
                                rst_root=args.rst_dir, batch_size=args.batch_size)
        '''
        if args.if_train:
            train_model.train(160001)
        else:
            train_model.test(args.input_dir)
        '''
        train_model.test(args.input_dir)
        
        while(True):
            with open("../../TestData/state.txt", "r") as state_file:
                line_list = state_file.readlines()
                if(len(line_list) == 0):
                    continue
                if(line_list[0] == "1\n"):
                    state_file.close()
                    train_model.test("../../TestData/1/")
                    try:
                        with open("../../TestData/state.txt", "w") as state_file:
                            state_file.write("0\n1\n0\n0")
                            state_file.close()
                    except PermissionError as e:
                        print("PermissionError")
                    # state_file.write("0")
                else:
                    state_file.close()

if __name__ == "__main__":
    main()
