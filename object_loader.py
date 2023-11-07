import os
from tqdm import tqdm
import pickle

# Local imports:
from common import Subject

PATH_TO_OBJECTS = os.path.join(os.curdir, "data", "serialized-objects")
PATH_TO_OUTPUT = os.path.join(os.curdir, "data")

obj_path_dict = {}  # key: subject id. value: path to the object file of the corresponding subject

# Find all object files and store their ids and paths:
for filename in os.listdir(PATH_TO_OBJECTS):
    if filename.endswith(".bin"):
        subject_id = int(filename[0:3])
        path_to_file = os.path.join(PATH_TO_OBJECTS, filename)
        obj_path_dict[subject_id] = path_to_file
        
def get_all_subjects(export_to_csvs=False) -> dict():
    # Load all objects to a dictionary
    subject_dict = {}
    for subject_id, obj_file in tqdm(obj_path_dict.items()):
        path_obj = os.path.join(PATH_TO_OUTPUT, "serialized-objects", str(subject_id).zfill(4) + ".bin")
        binary_file = open(path_obj, mode='rb')
        sub = pickle.load(binary_file)
        binary_file.close()
        subject_dict[subject_id] = sub

        if export_to_csvs:
            df = sub.export_to_dataframe()
            path_csv = os.path.join(PATH_TO_OUTPUT, "all-signals-cvs", subject_id.zfill(4), ".csv")
            df.to_csv(path_csv)

    return subject_dict

def get_subjects_by_id(first_subject_id: int, last_subject_id: int, export_to_csvs=False) -> dict():

    if first_subject_id < 1 or last_subject_id < first_subject_id:
        print(f"Illegal id range.")
        return None
    
    # Load objects to a dictionary
    subject_dict = {}
    for subject_id, obj_file in tqdm(obj_path_dict.items()):
        if first_subject_id<=subject_id <=last_subject_id:
            path_obj = os.path.join(PATH_TO_OUTPUT, "serialized-objects", str(subject_id).zfill(4) + ".bin")
            binary_file = open(path_obj, mode='rb')
            sub = pickle.load(binary_file)
            binary_file.close()
            subject_dict[subject_id] = sub

            if export_to_csvs:
                df = sub.export_to_dataframe()
                path_csv = os.path.join(PATH_TO_OUTPUT, "all-signals-cvs", subject_id.zfill(4), ".csv")
                df.to_csv(path_csv)

    return subject_dict