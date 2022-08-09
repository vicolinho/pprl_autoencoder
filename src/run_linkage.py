import argparse
import os
from termcolor import cprint
import json

from encode_scripts.lu_linking import link


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-cdir", type=str, default="./configs/N_A4", help="configuration directory")
    parser.add_argument('--single', default=False, action='store_true')
    args = parser.parse_args()
    config_dir = args.cdir
    single = args.single
    print('single encoder:',single)
    if not os.path.isdir(config_dir):
        cprint("config directory does not exist!","red")
    else:
        config_files = [f for f in os.listdir(config_dir) if os.path.isfile(os.path.join(config_dir, f)) and f.endswith(".json")]
        print(config_files)
        base_wd = os.getcwd()
        print(os.getcwd())
        for f in config_files:
            try:
                dirname = f.rstrip(".json")
                wd = os.path.join(config_dir,dirname)
                config = json.load(open(os.path.join(config_dir,f),"r"))
                os.chdir(wd)
                print(os.getcwd())
                if os.path.isfile('linking_results.txt'):
                    os.remove('linking_results.txt')
                if single:
                    link(config["encoder"]["path_encoded_a"],
                         config["encoder"]["path_encoded_b"],
                         config["encoder"]["path_meta_a"],
                         config["encoder"]["path_meta_b"],
                         None,
                         config["autoencoder"]["encoding_dimension"],
                         config["linking"]["linking_thresholds"],    
                         config["linking"]["result_logger_path"],
                         False)
                else:
                    link(config["encoder"]["path_encoded_a"],
                         config["encoder"]["path_encoded_b"],
                         config["encoder"]["path_meta_a"],
                         config["encoder"]["path_meta_b"],
                         config["mapper"]["save_path"],
                         config["autoencoder"]["encoding_dimension"],
                         config["linking"]["linking_thresholds"],
                         config["linking"]["result_logger_path"],
                         False)
                os.chdir(base_wd)
                print(os.getcwd())
            except Exception as e:
                cprint(e, "red")
