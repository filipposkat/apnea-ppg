import os
from tqdm import tqdm
import pickle
import yaml
from collections.abc import Generator
from pathlib import Path
import pandas as pd

# Local imports:
from common import Subject

# --- START OF CONSTANTS --- #
RELOAD_ANNOTATIONS = False
RELOAD_METADATA = True

with open("config.yml", 'r') as f:
    config = yaml.safe_load(f)

if config is not None:
    PATH_TO_OBJECTS = Path(config["paths"]["local"]["subject_objects_directory"])
    if "subject_metadata_file" in config["paths"]["local"]:
        PATH_TO_METADATA = Path(config["paths"]["local"]["subject_metadata_file"])
        PATH_TO_METADATA_NSSR = Path(config["paths"]["local"]["subject_metadata_nssr_file"])
    PATH_TO_ANNOTATIONS = Path(config["paths"]["local"]["xml_annotations_directory"])
    PATH_TO_OUTPUT = Path(config["paths"]["local"]["csv_directory"])
else:
    PATH_TO_OBJECTS = Path.cwd().joinpath("data", "serialized-objects")
    PATH_TO_METADATA = Path.cwd().joinpath("data", "mesa", "datasets", "mesa-sleep-dataset-0.7.0.csv")
    PATH_TO_METADATA_NSSR = Path.cwd().joinpath("data", "mesa", "datasets", "mesa-sleep-harmonized-dataset-0.7.0.csv")
    PATH_TO_ANNOTATIONS = Path.cwd().joinpath("data", "mesa", "polysomnography", "annotations-events-nsrr")
    PATH_TO_OUTPUT = Path.cwd().joinpath("data", "csvs")

# --- END OF CONSTANTS --- #

obj_path_dict = {}  # key: subject id. value: path to the object file of the corresponding subject
# Find all object files and store their ids and paths:
for filename in os.listdir(PATH_TO_OBJECTS):
    if filename.endswith(".bin"):
        subject_id = int(filename[0:4])
        path_to_file = os.path.join(PATH_TO_OBJECTS, filename)
        obj_path_dict[subject_id] = path_to_file


def get_all_ids():
    return obj_path_dict.keys()


def get_all_subjects(export_to_csvs=False) -> dict[int: Subject]:
    """
    :param export_to_csvs: if True then the subjects will be exported to csvs in the directory
    csv_directory/all-signals-csvs, where PATH_TO_OUTPUT is a constant defined in config.yml
    :return: dictionary {subject id: Subject}
    """
    # Load all objects to a dictionary
    subject_dict = {}
    for subject_id, obj_file in tqdm(obj_path_dict.items()):
        # path_obj = os.path.join(PATH_TO_OBJECTS, str(subject_id).zfill(4) + ".bin")
        binary_file = open(obj_file, mode='rb')
        sub = pickle.load(binary_file)
        binary_file.close()
        subject_dict[subject_id] = sub

        if export_to_csvs:
            df = sub.export_to_dataframe()
            path_csv = os.path.join(PATH_TO_OUTPUT, "all-signals-csvs", subject_id.zfill(4), ".csv")
            df.to_csv(path_csv)

    return subject_dict


def all_subjects_generator(progress_bar=True) -> Generator[tuple[int, Subject], None, None]:
    """
    Generator of Subjects
    :return: (int, Subject)
    """
    if progress_bar:
        for subject_id, obj_file in tqdm(obj_path_dict.items()):
            # path_obj = os.path.join(PATH_TO_OBJECTS, str(subject_id).zfill(4) + ".bin")
            binary_file = open(obj_file, mode='rb')
            sub = pickle.load(binary_file)
            binary_file.close()
            yield subject_id, sub
    else:
        for subject_id, obj_file in obj_path_dict.items():
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
    else:
        path_obj = os.path.join(PATH_TO_OBJECTS, str(subject_id).zfill(4) + ".bin")
        binary_file = open(path_obj, mode='rb')
        sub = pickle.load(binary_file)
        binary_file.close()

        if export_to_csvs:
            df = sub.export_to_dataframe()
            path_csv = os.path.join(PATH_TO_OUTPUT, "all-signals-cvs", str(subject_id).zfill(4), ".csv")
            df.to_csv(path_csv)

        return subject_id, sub


def get_subjects_by_id_range(first_subject_id: int, last_subject_id: int, export_to_csvs=False) -> dict[int: Subject]:
    """
    :param first_subject_id: The lower part of the desired id range
    :param last_subject_id: The upper part of the desired id range
    :param export_to_csvs: if True then the subjects will be exported to csvs in the directory
    csv_directory/all-signals-csvs, where PATH_TO_OUTPUT is a constant defined in config.yml
    """
    if first_subject_id < 1 or last_subject_id < first_subject_id:
        print(f"Illegal id range.")
        return None

    filtered_subject_paths = {id: obj_path_dict[id] for id in obj_path_dict.keys()
                              if first_subject_id <= id <= last_subject_id}

    # Load objects to a dictionary
    subject_dict = {}
    for subject_id, obj_file in tqdm(obj_path_dict.items()):
        # path_obj = os.path.join(PATH_TO_OBJECTS, str(subject_id).zfill(4) + ".bin")
        binary_file = open(obj_file, mode='rb')
        sub = pickle.load(binary_file)
        binary_file.close()
        subject_dict[subject_id] = sub

        if export_to_csvs:
            df = sub.export_to_dataframe()
            path_csv = os.path.join(PATH_TO_OUTPUT, "all-signals-cvs", str(subject_id.zfill(4)), ".csv")
            df.to_csv(path_csv)

    return subject_dict


def get_subjects_by_ids(subject_ids: list[int], export_to_csvs=False) -> dict[int: Subject]:
    """
    :param subject_ids: List with the desired subject ids
    :param export_to_csvs: if True then the subjects will be exported to csvs in the directory
    csv_directory/all-signals-csvs, where PATH_TO_OUTPUT is a constant defined in config.yml
    """
    if len(subject_ids) < 1:
        print(f"subjects_ids should contain at least one id.")
        return None

    filtered_subject_paths = {id: obj_path_dict[id] for id in subject_ids}

    # Load objects to a dictionary
    subject_dict = {}
    for id, obj_file in tqdm(filtered_subject_paths.items()):
        # path_obj = os.path.join(PATH_TO_OBJECTS, str(id).zfill(4) + ".bin")
        binary_file = open(obj_file, mode='rb')
        sub = pickle.load(binary_file)
        binary_file.close()
        subject_dict[id] = sub

        if export_to_csvs:
            df = sub.export_to_dataframe()
            path_csv = os.path.join(PATH_TO_OUTPUT, "all-signals-cvs", str(id.zfill(4)), ".csv")
            df.to_csv(path_csv)

    return subject_dict


def get_subjects_by_ids_generator(subject_ids: list[int], progress_bar=True) -> Generator[
    tuple[int, Subject], None, None]:
    """
    :param subject_ids: List with the desired subject ids
    """
    if len(subject_ids) < 1:
        print(f"subjects_ids should contain at least one id.")
        return None

    filtered_subject_paths = {id: obj_path_dict[id] for id in subject_ids}

    # Yield objects one by one
    if progress_bar:
        for id, obj_file in tqdm(filtered_subject_paths.items()):
            # path_obj = os.path.join(PATH_TO_OBJECTS, str(id).zfill(4) + ".bin")
            binary_file = open(obj_file, mode='rb')
            sub = pickle.load(binary_file)
            binary_file.close()
            yield id, sub
    else:
        for id, obj_file in filtered_subject_paths.items():
            # path_obj = os.path.join(PATH_TO_OBJECTS, str(id).zfill(4) + ".bin")
            binary_file = open(obj_file, mode='rb')
            sub = pickle.load(binary_file)
            binary_file.close()
            yield id, sub


if __name__ == "__main__":
    (id, sub) = get_subject_by_id(1212)
    print(sub.signal_headers)
    print(sub.metadata)
    # df = sub.export_to_dataframe(signal_labels=["Pleth"], max_frequency=32)
    # df.to_csv("107.csv")
    # print(df.shape[0])
    # print(sum(df["event_index"] == 1))
    # print(df.to_numpy())
    # print(sub.get_event_at_time(19290))

    # print(df["Pleth"].min())
    # print(df["Pleth"].max())
    # print(df.dtypes)
    # subs[133].export_to_dataframe()["Pleth"][646284:646909].plot()
    # plt.show()

    if RELOAD_METADATA:
        df = pd.read_csv(PATH_TO_METADATA, sep=',')
        df.set_index(keys="mesaid", drop=False, inplace=True)
        df.index.names = [None]

        if PATH_TO_METADATA_NSSR is not None:
            df_nssr = pd.read_csv(PATH_TO_METADATA_NSSR, sep=',')
            df_nssr.set_index(keys="mesaid", drop=False, inplace=True)
            df_nssr.index.names = [None]
            df_nssr.drop(["mesaid", "examnumber"], axis=1, inplace=True)
            df = pd.concat([df, df_nssr], axis=1, verify_integrity=True)

        for subject_id, sub in all_subjects_generator(progress_bar=True):
            sub_dict = df.loc[subject_id, :].to_dict()
            sub.import_metadata(sub_dict)

            path_obj = PATH_TO_OBJECTS.joinpath(str(subject_id).zfill(4) + ".bin")
            binary_file = open(path_obj, mode='wb')
            pickle.dump(sub, binary_file)
            binary_file.close()

    if RELOAD_ANNOTATIONS:
        annots_dict = {}  # key: subject id. value: path to the annotation xml file of the corresponding subject
        # Find all annotation xml files and store their ids and paths:
        for file in PATH_TO_ANNOTATIONS.iterdir():
            if file.name.endswith(".xml"):
                subject_id = int(file.name[-13:-9])
                annots_dict[subject_id] = file

        for subject_id, sub in all_subjects_generator():
            sub.import_annotations_from_xml(annots_dict[subject_id])
            # print(sub.export_to_dataframe())
            path_obj = PATH_TO_OBJECTS.joinpath(str(subject_id).zfill(4) + ".bin")
            binary_file = open(path_obj, mode='wb')
            pickle.dump(sub, binary_file)
            binary_file.close()
