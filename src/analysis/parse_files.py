import numpy as np
import pandas as pd

def parse_dataset_info(path):
    size_a = 0
    size_b = 0
    overlap = 0
    with open(path,"r") as info:
        for line in info:
            if "SIZE_A" in line:
                size_a = int(line.rstrip("\n").split(" ")[-1])
            if "SIZE_B" in line:
                size_b = int(line.rstrip("\n").split(" ")[-1])
            if "OVERLAP" in line:
                overlap = int(line.rstrip("\n").split(" ")[-1])
    return size_a,size_b,overlap

def parse_linking_results(path,threshold,ds="NCVR_0",tol=.001, hardening=""):
    linked = 0
    correct = 0
    lb,cb = False,False
    with open(path, "r") as f:
        found_threshold = False
        found_begin = False

        for line in f:
            if ds is not None:
                if "BEGIN "+ds+''+hardening in line:
                    found_begin = True
                if not found_begin:
                    continue
                if "END "+ds+''+hardening in line:
                    break
            if "threshold" in line:
                try:
                    thresh = float(line.rstrip("\n").split(" ")[-1])
                    if abs(threshold-thresh)<=tol:
                        found_threshold = True
                except Exception as e:
                    print(e)
            if not found_threshold:
                continue
            if "found" in line:
                linked = int(line.rstrip("\n").split(" ")[-1])
                lb = True
            if "correct" in line:
                correct = int(line.rstrip("\n").split(" ")[-1])
                cb = True
            if lb and cb:
                break
    return linked, correct



def parse_training_progress(path):
    df = pd.read_csv(path,sep=";")
    df["epoch"] = df["epoch"] + 1
    return df.values

def parse_training_times(path):
    df = pd.read_csv(path, names=["epoch","time"])
    print(df.head)
    return df.values
