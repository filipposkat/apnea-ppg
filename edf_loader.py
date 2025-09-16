import os
import yaml
from tqdm import tqdm
from pyedflib import highlevel
import pickle
from pathlib import Path
import pandas as pd

# Local imports:
from common import Subject

with open("config.yml", 'r') as f:
    config = yaml.safe_load(f)

if config is not None:
    PATH_TO_EDFS = config["paths"]["local"]["edf_directory"]
    PATH_TO_ANNOTATIONS = config["paths"]["local"]["xml_annotations_directory"]
    PATH_TO_METADATA = Path(config["paths"]["local"]["subject_metadata_file"])
    if "subject_metadata_nssr_file" in config["paths"]["local"]:
        PATH_TO_METADATA_NSRR = Path(config["paths"]["local"]["subject_metadata_nssr_file"])
    else:
        PATH_TO_METADATA_NSRR = None
    PATH_TO_OUTPUT = config["paths"]["local"]["subject_objects_directory"]
else:
    PATH_TO_EDFS = "D:\mesa\mesa\polysomnography\edfs"
    PATH_TO_ANNOTATIONS = "G:\\filip\Documents\Data Science Projects\Thesis\mesa\polysomnography\\annotations-events-nsrr"
    PATH_TO_METADATA = Path.cwd().joinpath("data", "mesa", "datasets", "mesa-sleep-dataset-0.6.0.csv")
    PATH_TO_METADATA_NSRR = Path.cwd().joinpath("data", "mesa", "datasets", "mesa-sleep-harmonized-dataset-0.7.0.csv")
    PATH_TO_OUTPUT = os.path.join(os.curdir, "data", "serialized-objects")

edf_dict = {}  # key: subject id. value: path to the edf of the corresponding subject
edf_list = []  # list with the paths of all edf files. Same as edf_dict.values()
# Find all edf files and store their ids and paths:
for filename in os.listdir(PATH_TO_EDFS):
    if filename.endswith(".edf"):
        subject_id = int(filename[-8:-4])
        path_to_file = os.path.join(PATH_TO_EDFS, filename)
        edf_dict[subject_id] = path_to_file
        edf_list.append(path_to_file)

signals, signal_headers, header = highlevel.read_edf(edf_list[0], ch_names=["Flow", "Snore", "Thor", "Pleth"])
# print(signal_headers)

annots_dict = {}  # key: subject id. value: path to the annotation xml file of the corresponding subject
annots_list = []  # list with the paths of all annotation xml files. Same as annots_dict.values()
# Find all annotation xml files and store their ids and paths:
for filename in os.listdir(PATH_TO_ANNOTATIONS):
    if filename.endswith(".xml"):
        subject_id = int(filename[-13:-9])
        path_to_file = os.path.join(PATH_TO_ANNOTATIONS, filename)
        annots_dict[subject_id] = path_to_file
        annots_list.append(path_to_file)

# sub = Subject(1)
# sub.import_signals_from_edf(edf_dict[1])
# sub.import_annotations_from_xml(annots_dict[1])
#
# df = sub.export_to_dataframe()
# print(df)

# Load metadata file:
df = pd.read_csv(PATH_TO_METADATA, sep=',')
df["mesaid"] = df["mesaid"].astype("int64")
df.set_index(keys="mesaid", drop=False, inplace=True)
df.index.names = [None]

if PATH_TO_METADATA_NSRR is not None:
    df_nssr = pd.read_csv(PATH_TO_METADATA_NSRR, sep=',')
    df_nssr["mesaid"] = df_nssr["mesaid"].astype("int64")
    df_nssr.set_index(keys="mesaid", drop=True, inplace=True)
    df_nssr.index.names = [None]
    df_nssr.drop("examnumber", axis=1, inplace=True)
    df = pd.concat([df, df_nssr], axis=1, verify_integrity=True)

print("Creating objects from edf files and serializing them as binary files:")
for subject_id, edf in tqdm(edf_dict.items()):
    sub = Subject(subject_id)
    sub.import_signals_from_edf(edf, channel_names=["Flow", "SpO2", "Pleth"])
    sub.import_annotations_from_xml(annots_dict[subject_id])
    sub_dict = df.loc[subject_id, :].to_dict()
    sub.import_metadata(sub_dict)

    path_obj = os.path.join(PATH_TO_OUTPUT, str(subject_id).zfill(4) + ".bin")
    binary_file = open(path_obj, mode='wb')
    pickle.dump(sub, binary_file)
    binary_file.close()

    # df = sub.export_to_dataframe()
    # path_csv = os.path.join(PATH_TO_OUTPUT, "all-signals-cvs", subject_id.zfill(4), ".csv")
    # df.to_csv(path_csv)
