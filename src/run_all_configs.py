import argparse
import json
import logging
import os
import datetime as dt
from termcolor import cprint

from encode_scripts.fit_encoder import fit_encoder
from encode_scripts.encode_data import encode
from encode_scripts.lu_data_generator import generate_data
from encode_scripts.b_encoding_mapper import decode_random
from encode_scripts.a_encoding_mapper import encode_decoded_random
from encode_scripts.lu_mapper_fit import fit_mapper
from encode_scripts.lu_linking import link

logging.basicConfig(filename="run_all.log", level=logging.INFO)

def run_all(config,cleanup=False):
    # fit a
    fit_encoder("model_a",
                config["autoencoder"]["input_dimension"],
                config["autoencoder"]["encoder_hidden_dimensions"],
                config["autoencoder"]["encoding_dimension"],
                config["autoencoder"]["decoder_hidden_dimensions"],
                config["autoencoder"]["output_dimension"],
                config["autoencoder"]["activation_function"],
                config["autoencoder"]["training_dataset"],
                "a",
                config["autoencoder"]["validation_split"],
                config["autoencoder"]["loss_function"],
                config["autoencoder"]["training_epochs"],
                config["autoencoder"]["batch_size"],
                config["autoencoder"]["save_path_a"],
                config["autoencoder"]["log_file_a"])

    # fit b
    fit_encoder("model_b",
                config["autoencoder"]["input_dimension"],
                config["autoencoder"]["encoder_hidden_dimensions"],
                config["autoencoder"]["encoding_dimension"],
                config["autoencoder"]["decoder_hidden_dimensions"],
                config["autoencoder"]["output_dimension"],
                config["autoencoder"]["activation_function"],
                config["autoencoder"]["training_dataset"],
                "b",
                config["autoencoder"]["validation_split"],
                config["autoencoder"]["loss_function"],
                config["autoencoder"]["training_epochs"],
                config["autoencoder"]["batch_size"],
                config["autoencoder"]["save_path_b"],
                config["autoencoder"]["log_file_b"])

    # encode a
    encode(config["encoder"]["encoding_dataset"],
           "a",
           config["autoencoder"]["save_path_a"],
           config["encoder"]["path_encoded_a"],
           config["encoder"]["path_meta_a"])

    # encode b
    encode(config["encoder"]["encoding_dataset"],
           "b",
           config["autoencoder"]["save_path_b"],
           config["encoder"]["path_encoded_b"],
           config["encoder"]["path_meta_b"])

    # genrate random data for mapper training
    generate_data(config["linking"]["training_data_length"],
                  config["autoencoder"]["encoding_dimension"],
                  config["linking"]["random_data_path"])

    # decode random data with decoder of model b
    decode_random(config["autoencoder"]["save_path_b"],
                  config["linking"]["random_data_path"],
                  config["encoder"]["path_meta_b"],
                  config["linking"]["decoded_data_path"])

    # encode decoded random data with encoder of model a
    encode_decoded_random(config["autoencoder"]["save_path_a"],
                          config["linking"]["decoded_data_path"],
                          config["encoder"]["path_meta_a"],
                          config["linking"]["encoded_data_path"])

    # fit encoding mapper
    fit_mapper(config["linking"]["random_data_path"],
               config["linking"]["encoded_data_path"],
               config["autoencoder"]["encoding_dimension"],
               config["mapper"]["hidden_dimensions"],
               config["mapper"]["loss_function"],
               config["mapper"]["validation_split"],
               config["mapper"]["training_epochs"],
               config["mapper"]["batch_size"],
               config["mapper"]["save_path"],
               config["mapper"]["logger_path"])

    # find matches in datasets
    link(config["encoder"]["path_encoded_a"],
         config["encoder"]["path_encoded_b"],
         config["encoder"]["path_meta_a"],
         config["encoder"]["path_meta_b"],
         config["mapper"]["save_path"],
         config["autoencoder"]["encoding_dimension"],
         config["linking"]["linking_thresholds"],
         config["linking"]["result_logger_path"],
         True)

    if cleanup:
        # clean up directory
        # os.remove(config["encoder"]["path_encoded_a"])
        # os.remove(config["encoder"]["path_encoded_b"])
        os.remove(config["linking"]["random_data_path"])
        os.remove(config["linking"]["decoded_data_path"])
        os.remove(config["linking"]["encoded_data_path"])


if __name__ == "__main__":
    logging.info("Starting new run: "+dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z"))
    parser = argparse.ArgumentParser()
    parser.add_argument("-cdir", type=str, default="./configs/N_A4", help="configuration directory")
    args = parser.parse_args()
    config_dir = args.cdir
    if not os.path.isdir(config_dir):
        cprint("config directory does not exist!", "red")
    else:
        config_files = [f for f in os.listdir(config_dir) if os.path.isfile(os.path.join(config_dir, f)) and
                        (f.endswith(".json"))]
        print(config_files)
        base_wd = os.getcwd()
        print(os.getcwd())
        for f in config_files:
            try:
                dirname = f.rstrip(".json")
                wd = os.path.join(config_dir,dirname)
                if os.path.isdir(wd):
                    cprint("skipping \""+f+"\", corresponding directory already exists!","red")
                    logging.info("skipped \""+f+"\", corresponding directory already exists!")
                    continue
                os.mkdir(wd)
                config = json.load(open(os.path.join(config_dir,f),"r"))
                os.chdir(wd)
                print(os.getcwd())
                run_all(config, cleanup=True)
                os.chdir(base_wd)
                print(os.getcwd())
            except Exception as e:
                cprint(e,"red")
                logging.error(e,exc_info=True)
                logging.info("using config file \""+f+"\"")
