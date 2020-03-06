from mcm.dataMoral import *
from mcm.funcs_mcm import mcm_template_quests
import csv
from tqdm import tqdm
import os
import argparse

parser = argparse.ArgumentParser(description='Crazy Stuff')
parser.add_argument('--model', default=None, type=str,
                    help='model name', required=True)

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)


if __name__ == '__main__':
    args = parser.parse_args()
    checkpoint_path = None
    if len(args.model.split('_')) > 1:
        model_name = args.model.replace(args.model.split('_')[0] + "_", "")
        checkpoint_path = 'retrain_use/skip_thoughts/trained_models/train_' + model_name + '/'

    experimental_quests_ = experimental_quests_paper

    while True:
        data = input("input action optional followed by context. Input <stop> to exit. \n")
        if data == "stop":
            exit()

        batch = [data]
        res = mcm_template_quests(experimental_quests_, batch, args.model, checkpoint_path)

        print(res[0][1], "has the score:", res[0][0])
