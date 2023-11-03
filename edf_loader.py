import os
from tqdm import tqdm
from pyedflib import highlevel
import pickle

# Local imports:
from common import Subject

PATH_TO_EDFS = "D:\mesa\mesa\polysomnography\edfs"
PATH_TO_ANNOTATIONS = "G:\\filip\Documents\Data Science Projects\Thesis\mesa\polysomnography\\annotations-events-nsrr"
PATH_TO_OUTPUT = os.path.join(os.curdir, "data")

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

print("Creating objects from edf files and serializing them as binary files:")
for subject_id, edf in tqdm(edf_dict.items()):
    sub = Subject(subject_id)
    sub.import_signals_from_edf(edf)
    sub.import_annotations_from_xml(annots_dict[subject_id])

    path_obj = os.path.join(PATH_TO_OUTPUT, "serialized-objects", str(subject_id).zfill(4) + ".bin")
    binary_file = open(path_obj, mode='wb')
    pickle.dump(sub, binary_file)
    binary_file.close()

    # df = sub.export_to_dataframe()
    # path_csv = os.path.join(PATH_TO_OUTPUT, "all-signals-cvs", subject_id.zfill(4), ".csv")
    # df.to_csv(path_csv)
