import os
from tqdm import tqdm
import pickle
import yaml
from collections.abc import Generator
import matplotlib.pyplot as plt
# Local imports:
from common import Subject

with open("config.yml", 'r') as f:
    config = yaml.safe_load(f)

if config is not None:
    PATH_TO_OBJECTS = config["paths"]["local"]["subject_objects_directory"]
    PATH_TO_OUTPUT = config["paths"]["local"]["csv_directory"]
else:
    PATH_TO_OBJECTS = os.path.join(os.curdir, "data", "serialized-objects")
    PATH_TO_OUTPUT = os.path.join(os.curdir, "data", "csvs")

obj_path_dict = {}  # key: subject id. value: path to the object file of the corresponding subject

# Find all object files and store their ids and paths:
for filename in os.listdir(PATH_TO_OBJECTS):
    if filename.endswith(".bin"):
        subject_id = int(filename[0:4])
        path_to_file = os.path.join(PATH_TO_OBJECTS, filename)
        obj_path_dict[subject_id] = path_to_file


def get_all_subjects(export_to_csvs=False) -> dict[int: Subject]:
    """
    :param export_to_csvs: if True then the subjects will be exported to csvs in the directory
    csv_directory/all-signals-csvs, where PATH_TO_OUTPUT is a constant defined in config.yml
    :return: dictionary {subject id: Subject}
    """
    # Load all objects to a dictionary
    subject_dict = {}
    for subject_id, obj_file in tqdm(obj_path_dict.items()):
        path_obj = os.path.join(PATH_TO_OBJECTS, str(subject_id).zfill(4) + ".bin")
        binary_file = open(path_obj, mode='rb')
        sub = pickle.load(binary_file)
        binary_file.close()
        subject_dict[subject_id] = sub

        if export_to_csvs:
            df = sub.export_to_dataframe()
            path_csv = os.path.join(PATH_TO_OUTPUT, "all-signals-csvs", subject_id.zfill(4), ".csv")
            df.to_csv(path_csv)

    return subject_dict


def all_subjects_generator() -> Generator[tuple[int, Subject], None, None]:
    """
    Generator of Subjects
    :return: (int, Subject)
    """
    for subject_id, obj_file in tqdm(obj_path_dict.items()):
        path_obj = os.path.join(PATH_TO_OBJECTS, str(subject_id).zfill(4) + ".bin")
        binary_file = open(path_obj, mode='rb')
        sub = pickle.load(binary_file)
        binary_file.close()
        yield subject_id, sub


def get_subject_by_id(subject_id: int, export_to_csvs=False) -> tuple[int: Subject]:
    """
    :param subject_id: Id of the desired subjected
    :param export_to_csvs: if True then the subjects will be exported to csvs in the directory
    csv_directory/all-signals-csvs, where PATH_TO_OUTPUT is a constant defined in config.yml
    """
    if subject_id < 1:
        print(f"Illegal id.")
        return None

    # Load objects to a dictionary
    subject_dict = {}
    for id, obj_file in obj_path_dict.items():
        if id == subject_id:
            path_obj = os.path.join(PATH_TO_OBJECTS, str(subject_id).zfill(4) + ".bin")
            binary_file = open(path_obj, mode='rb')
            sub = pickle.load(binary_file)
            binary_file.close()

            if export_to_csvs:
                df = sub.export_to_dataframe()
                path_csv = os.path.join(PATH_TO_OUTPUT, "all-signals-cvs", str(subject_id).zfill(4), ".csv")
                df.to_csv(path_csv)

            return subject_id, sub


def get_subjects_by_id(first_subject_id: int, last_subject_id: int, export_to_csvs=False) -> dict[int: Subject]:
    """
    :param export_to_csvs: if True then the subjects will be exported to csvs in the directory
    csv_directory/all-signals-csvs, where PATH_TO_OUTPUT is a constant defined in config.yml
    """
    if first_subject_id < 1 or last_subject_id < first_subject_id:
        print(f"Illegal id range.")
        return None

    # Load objects to a dictionary
    subject_dict = {}
    for subject_id, obj_file in tqdm(obj_path_dict.items()):
        if first_subject_id <= subject_id <= last_subject_id:
            path_obj = os.path.join(PATH_TO_OBJECTS, str(subject_id).zfill(4) + ".bin")
            binary_file = open(path_obj, mode='rb')
            sub = pickle.load(binary_file)
            binary_file.close()
            subject_dict[subject_id] = sub

            if export_to_csvs:
                df = sub.export_to_dataframe()
                path_csv = os.path.join(PATH_TO_OUTPUT, "all-signals-cvs", str(subject_id.zfill(4)), ".csv")
                df.to_csv(path_csv)

    return subject_dict


# subs = get_subjects_by_id(133, 133)
# print(subs[133].signal_headers)
# subs[133].export_to_dataframe()["Pleth"][646284:646909].plot()
# plt.show()

PATH_TO_ANNOTATIONS = "G:\\filip\Documents\Data Science Projects\Thesis\mesa\polysomnography\\annotations-events-nsrr"
annots_dict = {}  # key: subject id. value: path to the annotation xml file of the corresponding subject
annots_list = []  # list with the paths of all annotation xml files. Same as annots_dict.values()
# Find all annotation xml files and store their ids and paths:
for filename in os.listdir(PATH_TO_ANNOTATIONS):
    if filename.endswith(".xml"):
        subject_id = int(filename[-13:-9])
        path_to_file = os.path.join(PATH_TO_ANNOTATIONS, filename)
        annots_dict[subject_id] = path_to_file
        annots_list.append(path_to_file)

for subject_id, sub in all_subjects_generator():
    sub.import_annotations_from_xml(annots_dict[subject_id])

    path_obj = os.path.join(PATH_TO_OBJECTS, str(subject_id).zfill(4) + ".bin")
    binary_file = open(path_obj, mode='wb')
    pickle.dump(sub, binary_file)
    binary_file.close()
