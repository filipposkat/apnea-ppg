import numpy as np
import pandas as pd
import os
from itertools import islice
from tqdm import tqdm
from pyedflib import highlevel
from bs4 import BeautifulSoup

PATH_TO_EDFS = "D:\mesa\mesa\polysomnography\edfs"
PATH_TO_ANNOTATIONS =  "G:\\filip\Documents\Data Science Projects\Thesis\mesa\polysomnography\\annotations-events-nsrr"
PATH_TO_OUTPUT = os.path.join(os.curdir, "data")


def downsample_to_proportion(sequence, proportion: int) -> list:
    """Down-samples thr given sequence so that the returned sequence length is a proportion of the given sequence length

    :param sequence: Iterable
        Sequence to be down-sampled
    :param proportion: int
        Desired proportion of the output sequence length to the input sequence length

    :return: list
        The sequence down-sampled with length equal to (input sequence length * proportion)

    """
    return list(islice(sequence, 0, len(sequence), int(1 / proportion)))


class Subject:
    id: int
    signals: list[list]
    signal_headers: list[dict]
    start: int
    duration: float
    events: list[dict]

    def __init__(self, id):
        self.id = id

    def import_signals_from_edf(self, edf_file):
        # ch_names=["Flow", "Snore", "Thor", "Pleth"]
        signals, signal_headers, header = highlevel.read_edf(edf_file, ch_names=["Pleth"])
        self.signals = signals
        self.signal_headers = signal_headers

    def import_annotations_from_xml(self, xml_file):
        with open(xml_file, 'r') as f:
            data = f.read()
            bs_data = BeautifulSoup(data, "xml")

            self.start = int(bs_data.find("Start").string)
            self.duration = float(bs_data.find("Duration").string)

            bs_events = bs_data.findAll("ScoredEvent")
            relevant_events = []
            for bs_event in bs_events:
                event_type = bs_event.find("EventType").string
                event_concept = bs_event.find("EventConcept").string
                if "Central apnea" in event_concept:
                    relevant_events.append({"start": float(bs_event.find("Start").string),
                                            "duration": float(bs_event.find("Duration").string),
                                            "type": "respiratory",
                                            "concept": "central_apnea"})
                elif "Obstructive apnea" in event_concept:
                    relevant_events.append({"start": float(bs_event.find("Start").string),
                                            "duration": float(bs_event.find("Duration").string),
                                            "type": "respiratory",
                                            "concept": "obstructive_apnea"})
                elif "Hypopnea" in event_concept:
                    relevant_events.append({"start": float(bs_event.find("Start").string),
                                            "duration": float(bs_event.find("Duration").string),
                                            "type": "respiratory",
                                            "concept": "hypopnea"})
            self.events = relevant_events

    def get_event_at_time(self, time: float) -> int:
        """
        :param time: Time in seconds
        :return: 0=no events, 1=central apnea, 2=obstructive apnea, 3=hypopnea
        """
        for event in self.events:
            st = event["start"]
            fin = st + event["duration"]
            if st <= time <= fin:
                if event["concept"] == "central_apnea":
                    return 1
                elif event["concept"] == "obstructive_apnea":
                    return 2
                elif event["concept"] == "hypopnea":
                    return 3
            return 0

    def export_to_dataframe(self, signals_indices: list = None) -> pd.DataFrame:
        """
        Exports object to dataframe keeping only the specified signals.
        Indices:
        Flow 0
        Snore 1
        Thor 2
        Pleth 3

        :param signals_indices: list with indices of the signals that will be exported. If None then all signals are exported
        :return: Pandas dataframe
        """
        df = pd.DataFrame()
        if signals_indices is None:
            retained_signals = self.signals
            retained_signal_headers = self.signal_headers
        else:
            retained_signals = [self.signals[i] for i in signals_indices]
            retained_signal_headers = [self.signal_headers[i] for i in signals_indices]


        # First it is important to find which signal has the lowest frequency:
        min_freq_signal_header = min(retained_signal_headers, key=lambda h: float(h["sample_frequency"]))
        min_freq_signal_index = retained_signal_headers.index(min_freq_signal_header)
        min_freq = min_freq_signal_header["sample_frequency"]
        length = len(retained_signals[min_freq_signal_index])
        print(f"Signal label with minimum frequency ({min_freq}Hz): {min_freq_signal_header['label']}. "
              f"Down-sampling rest signals (if any), to match signal length: {length}")

        # Then the time column can be created
        time_seq = [i / min_freq for i in range(length)]
        df["time_secs"] = time_seq
        for i in range(len(retained_signals)):
            header = retained_signal_headers[i]
            freq = header["sample_frequency"]
            label = header["label"]
            proportion = min_freq / freq

            if proportion == 1.0:
                signal = retained_signals[i]
            else:
                signal = downsample_to_proportion(retained_signals[i], proportion)
            df[label] = signal
        return df


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


# for subject_id, edf in tqdm(edf_dict.items()):
#     sub = Subject(subject_id)
#     sub.import_signals_from_edf(edf)
#     sub.import_annotations_from_xml(annots_dict[subject_id])

sub = Subject(1)
sub.import_signals_from_edf(edf_dict[1])
sub.import_annotations_from_xml(annots_dict[1])

df = sub.export_to_dataframe()
print(df)