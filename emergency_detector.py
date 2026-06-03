import numpy as np
import librosa
import subprocess
import os
import cv2


class EmergencyDetector:

    SIREN_FREQ_LOW = 500
    SIREN_FREQ_HIGH = 2000
    SIREN_ENERGY_THRESHOLD = 0.35

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

        # once emergency is confirmed this stays True for the rest of the video
        # so we stop checking audio every single frame
        self.emergency_latched = False

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

    # samples 10 frames, extracts top 20% of each frame as sky region
    # converts to grayscale and measures average brightness
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

    # extracts the 1-second audio segment at this timestamp
    # checks if siren frequencies (500-2000 Hz) dominate more than 35% of total energy
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

    # counts matching blue pixels in the frame using HSV colour range
    # only runs at night - disabled for daytime because the sky triggers
    # too many false positives in the same blue HSV range
    def detect_blue_light(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.BLUE_LOW, self.BLUE_HIGH)
        blue_pixels = np.sum(mask > 0)
        return blue_pixels > self.BLUE_PIXEL_THRESHOLD

    # NOTE: vehicle behaviour is never used to trigger emergency_active
    # we are studying how behaviour differs between emergency and normal scenarios
    # if we used behaviour to define when the emergency is active, the thing
    # we are measuring would also be the thing we are detecting - circular logic
    def is_emergency_active(self, timestamp, frame, current_vehicles, prev_vehicles):
        # if emergency was already confirmed earlier in this video, keep it on
        # an ambulance does not turn its siren off and on mid-run
        # and the FFT check on every frame wastes time once we already know
        if self.emergency_latched:
            return True, ["siren"]

        siren = self.detect_siren(timestamp)

        # blue light only checked at night - see detect_blue_light comment above
        blue_light = self.detect_blue_light(frame) if self.daytime is False else False

        triggered_by = []
        if siren:
            triggered_by.append("siren")
        if blue_light:
            triggered_by.append("blue_light")

        # latch on first confirmed trigger so we never run FFT again
        if triggered_by:
            self.emergency_latched = True

        return len(triggered_by) > 0, triggered_by