import numpy as np
import librosa
import subprocess
import os
import cv2

class EmergencyDetector:

    SIREN_FREQ_LOW = 500
    SIREN_FREQ_HIGH = 2000
    SIREN_ENERGY_THRESHOLD = 0.35

    MIN_VEHICLES_MOVING_LATERALLY = 2
    LATERAL_MOVEMENT_THRESHOLD = 0.3

    BLUE_LOW = np.array([100, 150, 100])
    BLUE_HIGH = np.array([130, 255, 255])
    BLUE_PIXEL_THRESHOLD = 200

    def __init__(self, video_path):
        self.video_path = video_path
        self.audio_path = video_path.replace(".mp4", ".wav")
        self.audio = None
        self.sr = None
        self.daytime = None
        self._extract_audio()

    def _extract_audio(self):
        print("Extracting audio...")
        subprocess.run([
            "ffmpeg", "-i", self.video_path,
            "-ar", "22050", "-ac", "1",
            "-y", self.audio_path
        ], capture_output=True)
        if os.path.exists(self.audio_path):
            self.audio, self.sr = librosa.load(self.audio_path, sr=22050)
            print("Audio extracted successfully")
        else:
            print("Audio extraction failed")

    # Samples 10 frames, extracts top 20% of each frame as sky region
    # converts to grayscale and measures average brightness using NumPy
    # each frame votes day if sky > 120 and scene > 80 on 0-255 scale
    # majority vote across all frames determines day or night
    def detect_daytime(self, frames_sample):
        day_votes = 0
        for frame in frames_sample:
            h = frame.shape[0]
            sky = frame[:int(h * 0.2), :]
            sky_brightness = np.mean(cv2.cvtColor(sky, cv2.COLOR_BGR2GRAY))
            scene_brightness = np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            if sky_brightness > 120 and scene_brightness > 80:
                day_votes += 1
        is_day = day_votes > len(frames_sample) / 2
        print(f"Day/night: {day_votes}/{len(frames_sample)} day votes -> {'daytime' if is_day else 'nighttime'}")
        return is_day

    # extracts  audio segment
    # if siren frequencies dominate more than 35% of total energy - siren detected
    def detect_siren(self, timestamp):
        if self.audio is None:
            return False
        start = int(timestamp * self.sr)
        end = int((timestamp + 1) * self.sr)
        if end > len(self.audio):
            return False
        segment = self.audio[start:end]
        fft = np.abs(np.fft.rfft(segment))
        freqs = np.fft.rfftfreq(len(segment), 1 / self.sr)
        siren_mask = (freqs >= self.SIREN_FREQ_LOW) & (freqs <= self.SIREN_FREQ_HIGH)
        siren_energy = np.sum(fft[siren_mask])
        total_energy = np.sum(fft)
        if total_energy == 0:
            return False
        ratio = siren_energy / total_energy
        return ratio > self.SIREN_ENERGY_THRESHOLD

    # counts matching blue pixels in the frame
    # only used for nighttime videos - disabled for daytime to avoid sky false positives
    def detect_blue_light(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.BLUE_LOW, self.BLUE_HIGH)
        blue_pixels = np.sum(mask > 0)
        return blue_pixels > self.BLUE_PIXEL_THRESHOLD

    def detect_vehicle_behaviour(self, current_vehicles, prev_vehicles):
        """"
        Detects yielding behaviour per vehicle.
        NOTE: This method is NOT used for emergency_active triggering

        Rule 1: lateral change > 0.5m in one second (sudden fast pullover)
        Rule 2: heading angle > 15 degrees (car clearly angled away)
        Rule 3: cumulative lateral movement > 0.8m in same direction over 3s
        Rule 4: in dev
        """""
        if not prev_vehicles:
            return False

        prev_dict = {v["track_id"]: v for v in prev_vehicles}
        yielding_vehicles = 0

        for v in current_vehicles:
            tid = v["track_id"]
            curr_lateral = v.get("lateral_offset", 0.0)
            curr_heading = abs(v.get("heading_angle", 0.0))

            # update lateral history
            if tid not in self.lateral_history:
                self.lateral_history[tid] = []
            self.lateral_history[tid].append(curr_lateral)
            if len(self.lateral_history[tid]) > self.CUMULATIVE_WINDOW:
                self.lateral_history[tid].pop(0)

            yielding = False

            # Rule 1 - sudden fast lateral movement > 0.5m in one second
            if tid in prev_dict:
                prev_lateral = prev_dict[tid].get("lateral_offset", 0.0)
                if abs(curr_lateral - prev_lateral) >= 0.5:
                    yielding = True

            # Rule 2 - car clearly angled away, heading > 15 degrees
            if curr_heading >= self.HEADING_ANGLE_THRESHOLD:
                yielding = True

            # Rule 3 - cumulative lateral displacement > 0.8m over 3 seconds
            # must be consistent direction (not oscillating)
            if len(self.lateral_history[tid]) >= self.CUMULATIVE_WINDOW:
                history = self.lateral_history[tid]
                total_displacement = abs(history[-1] - history[0])
                direction = history[-1] - history[0]
                consistent = all(
                    (history[i + 1] - history[i]) * direction >= 0
                    for i in range(len(history) - 1)
                )
                if total_displacement >= self.CUMULATIVE_LATERAL_THRESHOLD and consistent:
                    yielding = True

            if yielding:
                yielding_vehicles += 1

        return yielding_vehicles >= self.MIN_VEHICLES_MOVING_LATERALLY

    """"
    The emergency trigger must be independent of the vehicles reactions/behaviour
    we are comparing behaviour WITH vs WITHOUT emergency vehicle. If we use behaviour
    to define when the emergency is active,analysis becomes circular
    measuring what is already assumed.
    """

    def is_emergency_active(self, timestamp, frame, current_vehicles, prev_vehicles):
        siren = self.detect_siren(timestamp)
        blue_light = self.detect_blue_light(frame) if self.daytime is False else False

        triggered_by = []
        if siren:
            triggered_by.append("siren")
        if blue_light:
            triggered_by.append("blue_light")

        return len(triggered_by) > 0, triggered_by