import numpy as np
from scipy.signal import butter, filtfilt, upfirdn, decimate, firwin, resample_poly, get_window
import pandas as pd
from itertools import islice
from pathlib import Path


def detect_desaturations_profusion(spo2_values, sampling_rate, min_drop=3, max_fall_rate=4, max_plateau=60,
                                   max_drop_threshold=50, min_event_duration=1, max_event_duration=None):
    desaturation_events = 0
    min_event_samples = min_event_duration * sampling_rate
    max_event_samples = max_event_duration * sampling_rate if max_event_duration else len(spo2_values)
    max_plateau_samples = max_plateau * sampling_rate
    i = 0
    while i < len(spo2_values) - 1:
        # Look for a local zenith
        while i < len(spo2_values) - 1 and (spo2_values[i + 1] >= spo2_values[i] or spo2_values[i] > 100):
            i += 1
        zenith = spo2_values[i]

        # Start looking for a desaturation event
        start = i
        while i < len(spo2_values) - 1 and (spo2_values[i + 1] <= spo2_values[i] or spo2_values[i] < 50):
            i += 1

        nadir = spo2_values[i]
        for j in range(i + 1, i + max_plateau_samples):
            if j >= len(spo2_values):
                break
            val = spo2_values[j]
            if 100 > val > zenith:
                break
            elif nadir >= val > 50:
                nadir = val
                i = j

        event_length = i - start
        if min_event_samples <= event_length <= max_event_samples:
            drop = zenith - spo2_values[i]
            duration_seconds = event_length / sampling_rate

            # Calculate drop rate
            drop_rate = drop / duration_seconds

            # Check desaturation criteria
            if min_drop <= drop <= max_drop_threshold and drop_rate <= max_fall_rate:
                desaturation_events += 1

    return desaturation_events


def detect_desaturations_simple(spo2_values, min_length=3, max_duration_samples=120):
    """
    Calculate the number of 3% desaturations in a sequence of SpO2 values,
    based on the local zenith (highest point) and a minimum event length.

    Parameters:
    - spo2_values: A 1D numpy array of SpO2 values.
    - min_length: The minimum number of consecutive samples that the desaturation must last.

    Returns:
    - desaturation_count: The number of desaturation events that meet the criteria.
    """

    # Initialize variables
    desaturation_count = 0
    event_length = 0
    plateu = 0
    in_desaturation = False

    # Start with the first value as the initial local zenith (highest point before desaturation)
    local_zenith = spo2_values[0]

    # Iterate through the SpO2 values
    for i in range(1, len(spo2_values)):
        current_value = spo2_values[i]

        # if current_value < 50:
        #     continue

        # Check if the current value is part of a desaturation event (3% below local zenith)
        if current_value <= local_zenith - 3:
            event_length += 1
            in_desaturation = True

            if event_length > 1:
                mean_over_event = np.mean(spo2_values[i - event_length:i])
                if np.abs(current_value - mean_over_event) <= 1:
                    plateu += 1

            # Check if the maximum desaturation length is exceeded
            if max_duration_samples is not None and event_length > max_duration_samples:
                # Reset the event if it exceeds the maximum allowed length
                in_desaturation = False
                event_length = 0
        else:
            # If a desaturation event ends, check if it lasted long enough
            if in_desaturation and event_length >= min_length:
                desaturation_count += 1

            # Reset event tracking
            event_length = 0
            in_desaturation = False

            # Update the local zenith to the current value if it is higher than the previous local zenith
            if current_value > local_zenith:
                local_zenith = current_value

    # Final check in case the desaturation event ends at the last value
    if in_desaturation and event_length >= min_length:
        desaturation_count += 1

    return desaturation_count


def detect_desaturations_legacy(spo2_values, sampling_rate=1, drop_threshold=3, window_seconds=120,
                                min_duration_seconds=10, max_duration=120):
    # Convert window and duration parameters from seconds to number of samples
    window_samples = int(window_seconds * sampling_rate)
    min_duration_samples = int(min_duration_seconds * sampling_rate)
    max_duration_samples = int(max_duration * sampling_rate)
    desaturation_count = 0
    event_length = 0
    in_desaturation = False

    # Iterate through the SpO2 values, starting after the initial window
    for i in range(window_samples, len(spo2_values)):
        # Calculate the mean SpO2 over the last 'window_samples'
        mean_spo2 = np.mean(spo2_values[i - window_samples:i])

        # Check if the current value is at least 'drop_threshold' below the mean SpO2
        if spo2_values[i] <= mean_spo2 - drop_threshold:
            event_length += 1
            in_desaturation = True

            # Check if the maximum desaturation length is exceeded
            if max_duration_samples is not None and event_length > max_duration_samples:
                # Reset the event if it exceeds the maximum allowed length
                desaturation_count += 1
                in_desaturation = False
                event_length = 0
        else:
            # If a desaturation event ends, check if it lasted long enough
            if in_desaturation and event_length >= min_duration_samples:
                desaturation_count += 1

            # Reset event tracking
            event_length = 0
            in_desaturation = False

    # Final check in case the desaturation event ends at the last value
    if in_desaturation and event_length >= min_duration_samples:
        desaturation_count += 1

    return desaturation_count


def downsample_to_proportion(sequence, proportion: int, lpf=True) -> list | np.ndarray:
    """Down-samples the given sequence so that the returned sequence length is a proportion of the given sequence length

    :param sequence: Iterable
        Sequence to be down-sampled
    :param proportion: int
        Desired ratio of the output sequence length to the input sequence length

    :return: list
        The sequence down-sampled with length equal to (input sequence length * proportion)

    """
    if lpf:
        # # Calculate downsampling factor
        # downsample_factor = 1 / proportion

        # # Design a low-pass Butterworth filter
        # order = 4  # Filter order
        # b, a = butter(order, proportion, btype='low', output="ba", analog=False)  # ba returns IIR filter

        # # Apply the filter
        # sequence = filtfilt(b, a, sequence)

        expected_len = int(len(sequence) * proportion)
        downsample_factor = int(np.ceil(1 / proportion))
        N = downsample_factor * 32
        downsampled_signal = decimate(x=sequence, q=downsample_factor, n=N,
                                      ftype="fir")  # By default, Hamming window is used
        return downsampled_signal[:expected_len]
    else:
        return list(islice(sequence, 0, len(sequence), int(1 / proportion)))


def upsample_to_proportion(sequence, proportion: int) -> np.ndarray:
    """Up-samples the given sequence so that the returned sequence length is a proportion of the given sequence length

    :param sequence: Iterable
        Sequence to be up-sampled
    :param proportion: int
        Desired proportion of the output sequence length to the input sequence length

    :return: list
        The sequence up-sampled with length equal to (input sequence length * proportion)

    """
    # print(f"US1. Original length {len(sequence)}. Prop: {proportion}")
    # # Upsample by inserting zeros
    # upsampled_signal = upfirdn([1], sequence, up=proportion)  

    # # Design a low-pass Butterworth filter
    # order = 4  # Filter order
    # b, a = butter(order, 1 / proportion, btype='low', output="sos", analog=False)  # sos returns FIR filter

    # if lpf:
    #     # Apply the filter to smooth the upsampled signal
    #     upsampled_signal = filtfilt(b, a, upsampled_signal)

    N = int(32 * proportion) + 1
    window = get_window(window="hamming", Nx=N, fftbins=False)
    upsampled_signal = resample_poly(x=sequence, up=proportion, down=1, window=window, padtype="median")
    # upsampled_signal = resample_poly(x=sequence, up=proportion, down=1, padtype="median")
    return upsampled_signal


class Subject:
    id: int
    signals: list[np.ndarray]
    signal_headers: list[dict]
    start: int
    duration: float
    respiratory_events: list[dict]
    metadata: dict

    def __init__(self, id):
        self.id = id

    def import_metadata(self, metadata_dict: dict):
        self.metadata = metadata_dict

    def import_signals_from_edf(self, edf_file, channel_names=("Flow", "SpO2" "Pleth")):
        from pyedflib import highlevel
        # ch_names=["Flow", "Snore", "Thor", "Pleth"]
        signals, signal_headers, header = highlevel.read_edf(edf_file, ch_names=channel_names)
        self.signals = signals
        self.signal_headers = signal_headers

    def import_annotations_from_xml(self, xml_file):
        """
        Imports all respiratory event annotations and saves them in the events attribute of the object. If events object is already
        :param xml_file: Path to xml annotations file
        :return:
        """
        from bs4 import BeautifulSoup
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

            # Sort events by start time:
            self.respiratory_events = sorted(resp_events, key=lambda evnt: evnt["start"])

    def get_events_by_concept(self, concept: str) -> list[dict[str: str | float]]:
        """
        :param concept: Name of the event concept (e.g. central_apnea, obstructive_apnea, hypopnea, spo2_desat, etc.)
        :return: List with all the events with the specified concept
        """
        return [event for event in self.respiratory_events if event["concept"] == concept]

    def get_event_at_time(self, time: float) -> int:
        """
        :param time: Time in seconds
        :return: 0=no events, 1=central apnea, 2=obstructive apnea, 3=hypopnea, 4=spO2 desaturation
        """
        concepts_at_t = []
        for event in self.respiratory_events:
            st = event["start"]
            duration = event["duration"]
            fin = st + duration
            if st <= time <= fin:
                if event["concept"] == "central_apnea":
                    concepts_at_t.append(1)
                elif event["concept"] == "obstructive_apnea":
                    concepts_at_t.append(2)
                elif event["concept"] == "hypopnea":
                    concepts_at_t.append(3)
                elif event["concept"] == "spo2_desat":
                    concepts_at_t.append(4)
                # else:
                #     # Other respiratory event
                #     return 5
                elif st > time:
                    # If start time of event is bigger than time then there is no need to check more events
                    # because they are sorted by start time
                    break

        if len(concepts_at_t) > 0:
            # Give priority to central_apnea then obstructive then hypopnea then spo2_desat
            return min(concepts_at_t)
        else:
            # No event found
            return 0

    def assign_annotations_to_time_series(self, time_seq: pd.Series) -> pd.Series:
        """
        :param time_seq: Time series in seconds
        :return: Series with annotations corresponding to time,
        where: 0=no events, 1=central apnea, 2=obstructive apnea, 3=hypopnea, 4=spO2 desaturation
        """
        central_apnea_events = self.get_events_by_concept("central_apnea")
        obstructive_apnea_events = self.get_events_by_concept("obstructive_apnea")
        hypopnea_events = self.get_events_by_concept("hypopnea")
        spo2_desat_events = self.get_events_by_concept("spo2_desat")

        initial = np.zeros(len(time_seq))
        annotations = pd.Series(initial, index=time_seq.index, dtype="uint8")
        # Assign event concepts with this order. First concepts have the lowest priority and last concepts the highest
        # priority (they override previous annotations):
        for e in spo2_desat_events:
            s = e["start"]
            d = e["duration"]
            f = s + d
            annotations[(s <= time_seq) & (time_seq <= f)] = 4

        for e in hypopnea_events:
            s = e["start"]
            d = e["duration"]
            f = s + d
            annotations[(s <= time_seq) & (time_seq <= f)] = 3

        for e in obstructive_apnea_events:
            s = e["start"]
            d = e["duration"]
            f = s + d
            annotations[(s <= time_seq) & (time_seq <= f)] = 2

        for e in central_apnea_events:
            s = e["start"]
            d = e["duration"]
            f = s + d
            annotations[(s <= time_seq) & (time_seq <= f)] = 1
        return annotations

    def export_to_dataframe(self, signal_labels: list[str] = None, frequency: float = None,
                            print_downsampling_details=True, anti_aliasing=True, trim_signals=True,
                            median_to_low_spo2_values=True) -> pd.DataFrame:
        """
        Exports object to dataframe keeping only the specified signals.

        :param frequency: Desired frequency. If all signals will be resampled to this frequency
        :param print_downsampling_details: Whether to print details about the down-sampling performed to match signal
        lengths
        :param signal_labels: list with the names of the signals that will be exported. If None then all signals are
        exported.
        :param anti_aliasing: if True anti-aliasing LPF filter will be used when downsampling (for upsampling lpf will always be used).
        :param trim_signals: if True Min frequency signal will be zero trimmed and all other signals will be adjusted
        accordingly.
        :param median_to_low_spo2_values: After trimming, Values in SpO2 below 60, are replaced with the median of all SpO2 values
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

        if trim_signals:
            mfs = retained_signals[min_freq_signal_index]  # Min Frequency Signal
            original_mfs_length = len(mfs)
            # Trim zeros from front:
            mfs = np.trim_zeros(mfs, trim='f')
            front_zeros = original_mfs_length - len(mfs)

            # Trim zeros from back:
            tmp_len = len(mfs)
            mfs = np.trim_zeros(mfs, trim='b')
            back_zeros = tmp_len - len(mfs)

            retained_signals[min_freq_signal_index] = mfs

            # Adjust the other signals accordingly:
            for i in range(len(retained_signals)):
                if i != min_freq_signal_index:
                    freq = retained_signal_headers[i]["sample_frequency"]
                    assert freq % min_freq == 0
                    proportion = freq // min_freq
                    assert proportion == len(retained_signals[i]) // original_mfs_length
                    front_zeros_to_drop = front_zeros * int(proportion)
                    back_zeros_to_drop = back_zeros * int(proportion)
                    if back_zeros_to_drop == 0:
                        retained_signals[i] = retained_signals[i][front_zeros_to_drop:]
                        # if front_zeros_to_drop == 0:
                        #     print(f"{self.id}, ")
                    else:
                        retained_signals[i] = retained_signals[i][front_zeros_to_drop:-back_zeros_to_drop]
                    assert proportion == len(retained_signals[i]) / len(mfs)

        # # SpO2 may be incomplete: at start and end may have zeros:
        # if trim_spo2 and "SpO2" in signal_labels:
        #     spo2_i = 0
        #     for i in range(len(retained_signal_headers)):
        #         if retained_signal_headers[i]["label"] == "SpO2":
        #             spo2_i = i
        #             break
        #
        #     spo2 = retained_signals[spo2_i]
        #     original_spo2_length = len(spo2)
        #     # Trim zeros from front:
        #     spo2 = np.trim_zeros(spo2, trim='f')
        #     front_zeros = original_spo2_length - len(spo2)
        #
        #     # Trim zeros from back:
        #     tmp_len = len(spo2)
        #     spo2 = np.trim_zeros(spo2, trim='b')
        #     back_zeros = tmp_len - len(spo2)
        #
        #     retained_signals[spo2_i] = spo2
        #
        #     # Adjust the other signals accordingly:
        #     spo2_f = retained_signal_headers[spo2_i]["sample_frequency"]
        #     for i in range(len(retained_signals)):
        #         if i != spo2_i:
        #             freq = retained_signal_headers[i]["sample_frequency"]
        #             assert freq % spo2_f == 0
        #             proportion = freq // spo2_f  # SpO2 has 1Hz frequency so proportion is always > 1
        #             assert proportion == len(retained_signals[i]) // original_spo2_length
        #             front_zeros_to_drop = front_zeros * int(proportion)
        #             back_zeros_to_drop = back_zeros * int(proportion)
        #             if back_zeros_to_drop == 0:
        #                 retained_signals[i] = retained_signals[i][front_zeros_to_drop:]
        #                 if front_zeros_to_drop == 0:
        #                     print(f"{self.id}, ")
        #             else:
        #                 retained_signals[i] = retained_signals[i][front_zeros_to_drop:-back_zeros_to_drop]
        #             assert proportion == len(retained_signals[i]) / len(spo2)

        if median_to_low_spo2_values:
            threshold = 60
            spo2_i = -1
            for i in range(len(retained_signals)):
                lbl = retained_signal_headers[i]["label"]
                if lbl == "SpO2":
                    spo2_i = i
                    break
            if spo2_i != -1:
                spo2 = retained_signals[spo2_i]
                median_spo2_value = np.median(spo2)
                retained_signals[spo2_i][spo2 <= threshold] = median_spo2_value

        if frequency is not None:
            if print_downsampling_details:
                print(f"All signals will be resampled to ({frequency}Hz)/ "
                      f"Resampling-sampling all signals. Anti-aliasing: {anti_aliasing}.")
        else:
            frequency = min_freq
            length = len(retained_signals[min_freq_signal_index])
            if print_downsampling_details:
                print(f"Signal label with minimum frequency ({min_freq}Hz): {min_freq_signal_header['label']}. "
                      f"Down-sampling rest signals (if any), to match signal length: {length}. Anti-aliasing: {anti_aliasing}.")

        for i in range(len(retained_signals)):
            header = retained_signal_headers[i]
            freq = header["sample_frequency"]
            label = header["label"]
            proportion = frequency / freq

            if proportion == 1.0:
                signal = retained_signals[i]
            elif proportion < 1.0:
                signal = downsample_to_proportion(retained_signals[i], proportion, lpf=anti_aliasing)
            else:
                signal = upsample_to_proportion(retained_signals[i], proportion)

            df[label] = signal
            df[label] = df[label].astype("float32")  # Set type to 32 bit instead of 64 to save memory

        # Then the time column can be created
        time_seq = [i / frequency for i in range(df.shape[0])]
        df["time_secs"] = time_seq
        df["time_secs"] = df["time_secs"].astype("float32")  # Save memory

        # df["event_index"] = df["time_secs"].map(lambda t: self.get_event_at_time(t)).astype("uint8")
        df["event_index"] = self.assign_annotations_to_time_series(df["time_secs"])

        # df = df.astype({"time_secs": "float32"})  # Save memory
        return df
