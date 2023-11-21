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
    respiratory_events: list[dict]

    def __init__(self, id):
        self.id = id

    def import_signals_from_edf(self, edf_file):
        # ch_names=["Flow", "Snore", "Thor", "Pleth"]
        signals, signal_headers, header = highlevel.read_edf(edf_file, ch_names=["Flow", "Pleth"])
        self.signals = signals
        self.signal_headers = signal_headers

    def import_annotations_from_xml(self, xml_file):
        """
        Imports all respiratory event annotations and saves them in the events attribute of the object. If events object is already
        :param xml_file: Path to xml annotations file
        :return:
        """
        with open(xml_file, 'r') as f:
            data = f.read()
            bs_data = BeautifulSoup(data, "xml")

            self.start = int(bs_data.find("Start").string)
            self.duration = float(bs_data.find("Duration").string)

            bs_events = bs_data.findAll("ScoredEvent")
            resp_events = []
            for bs_event in bs_events:
                event_type = bs_event.find("EventType").string
                event_concept = bs_event.find("EventConcept").string
                if event_type is None:
                    continue

                if "Respiratory" in event_type:
                    if "Central apnea" in event_concept:
                        resp_events.append({"start": float(bs_event.find("Start").string),
                                            "duration": float(bs_event.find("Duration").string),
                                            "type": "respiratory",
                                            "concept": "central_apnea"})
                    elif "Obstructive apnea" in event_concept:
                        resp_events.append({"start": float(bs_event.find("Start").string),
                                            "duration": float(bs_event.find("Duration").string),
                                            "type": "respiratory",
                                            "concept": "obstructive_apnea"})
                    elif "Hypopnea" in event_concept:
                        resp_events.append({"start": float(bs_event.find("Start").string),
                                            "duration": float(bs_event.find("Duration").string),
                                            "type": "respiratory",
                                            "concept": "hypopnea"})
                    elif "SpO2 desaturation" in event_concept:
                        resp_events.append({"start": float(bs_event.find("Start").string),
                                            "duration": float(bs_event.find("Duration").string),
                                            "type": "respiratory",
                                            "concept": "spo2_desat"})
                    else:
                        resp_events.append({"start": float(bs_event.find("Start").string),
                                            "duration": float(bs_event.find("Duration").string),
                                            "type": "respiratory",
                                            "concept": bs_event.find("EventConcept").string.replace(' ', '_').lower()})

            self.respiratory_events = resp_events

    def get_events_by_concept(self, concept: str) -> list[dict[str: str | float]]:
        """
        :param concept: Name of the event concept (e.g. central_apnea, obstructive_apnea, hypopnea, spo2_desat, etc.)
        :return: List with all the events with the specified concept
        """
        return [event for event in self.respiratory_events if event["concept"] == concept]

    def get_event_at_time(self, time: float) -> int:
        """
        :param time: Time in seconds
        :return: 0=no events, 1=central apnea, 2=obstructive apnea, 3=hypopnea, 4=spO2 desaturation, 5=other event
        """
        for event in self.respiratory_events:
            st = event["start"]
            fin = st + event["duration"]
            if st <= time <= fin:
                if event["concept"] == "central_apnea":
                    return 1
                elif event["concept"] == "obstructive_apnea":
                    return 2
                elif event["concept"] == "hypopnea":
                    return 3
                elif event["concept"] == "spo2_desat":
                    return 4
                else:
                    # Other respiratory event
                    return 5
            return 0

    def export_to_dataframe(self, signal_labels: list[str] = None, print_downsampling_details=True) -> pd.DataFrame:
        """
        Exports object to dataframe keeping only the specified signals.

        :param signal_labels: list with the names of the signals that will be exported. If None then all signals are
        exported.
        :return: Pandas dataframe
        """
        df = pd.DataFrame()
        if signal_labels is None:
            retained_signals = self.signals
            retained_signal_headers = self.signal_headers
        else:
            retained_signals = [self.signals[i] for i in range(len(self.signals))
                                if self.signal_headers[i]["label"] in signal_labels]
            retained_signal_headers = [self.signal_headers[i] for i in range(len(self.signals))
                                       if self.signal_headers[i]["label"] in signal_labels]

        # First it is important to find which signal has the lowest frequency:
        min_freq_signal_header = min(retained_signal_headers, key=lambda h: float(h["sample_frequency"]))
        min_freq_signal_index = retained_signal_headers.index(min_freq_signal_header)
        min_freq = min_freq_signal_header["sample_frequency"]
        length = len(retained_signals[min_freq_signal_index])
        if print_downsampling_details:
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
