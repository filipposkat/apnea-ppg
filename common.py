import numpy as np
import pandas as pd
from itertools import islice
from bs4 import BeautifulSoup
from pyedflib import highlevel


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
        signals, signal_headers, header = highlevel.read_edf(edf_file, ch_names=["Flow", "Pleth"])
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

        df["event_index"] = df["time_secs"].map(lambda t: self.get_event_at_time(t))
        return df
