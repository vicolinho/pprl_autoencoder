from data_processing.data_sources import DATA_PATHS

# script for splitting NCVR csv-file into separate files for two data owners "A" and "B"

export_path = "/".join(DATA_PATHS["NCVR_raw"].split("/")[:-1])
print(export_path)

with open(DATA_PATHS["NCVR_raw"],"r") as csv, open(export_path+"/NCVR_A.csv","w") as data_A, open(export_path+"/NCVR_B.csv","w") as data_B:
    for row in csv:
        row_split = row.split(",")
        if row_split[1] == "A":
            row_split.pop(1)
            data_A.write(",".join(row_split))
        elif row_split[1] == "B":
            row_split.pop(1)
            data_B.write(",".join(row_split))
        else:
            print("cant assign row:")
            print(row)
