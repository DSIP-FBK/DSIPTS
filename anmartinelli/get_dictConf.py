import pickle as pkl
import argparse
from conf_dicts import dictConfiguration

def get_dict_conf(paths_pkl):
    for file_pkl in paths_pkl:
        with open(file_pkl, 'rb') as f: #! PATH for PKL
            dictConf, _  = pkl.load(f)
            f.close()
        print('='*20)
        print(f'\nFILE NAME: {file_pkl}')
        print(dictConf)

if __name__=='__main__':
    parser = argparse.ArgumentParser(description="Print Models' DictConfs")
    parser.add_argument("-p", "--paths", type=str, nargs='+', help="list of Models' paths for parameters inspection")
    args = parser.parse_args()

    get_dict_conf(args.paths)