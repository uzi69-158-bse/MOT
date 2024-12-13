from collections import deque
import numpy as np
import torch

class DeepSort:
    def __init__(self, config=None):
        self.trackers = []  # Placeholder for your tracker objects
        self.max_age = config['max_age']
        self.n_init = config['n_init']
        self.tracks = []  # List to hold current tracks
        self.frame_count = 0  # To keep track of the number of frames processed

    def update(self, detections):
        """Update tracker with new detections."""
        tracked_objects = []
        self.frame_count += 1

        # Update existing tracks
        for track in self.tracks:
            if track['age'] >= self.max_age:
                continue  # Remove old tracks
            track['age'] += 1  # Increment age

        for detection in detections:
            obj_id = detection.get('id')  # Ensure 'id' is present
            label = detection.get('label', 'Unknown')  # Default to 'Unknown' if 'label' is not present

            # Assuming you have logic to associate detections with existing tracks
            existing_track = next((t for t in self.tracks if t['id'] == obj_id), None)
            if existing_track:
                # Update existing track with new detection data
                existing_track['bbox'] = detection['bbox']  # Update bbox
                existing_track['label'] = label  # Update label
                existing_track['confidence'] = detection.get('confidence', 0)  # Update confidence
                tracked_objects.append(existing_track)
            else:
                # Add new track
                new_track = {
                    'id': obj_id,
                    'bbox': detection['bbox'],
                    'label': label,
                    'confidence': detection.get('confidence', 0),
                    'age': 1,  # Initialize age
                }
                self.tracks.append(new_track)
                tracked_objects.append(new_track)

        # Clean up old tracks
        self.tracks = [track for track in self.tracks if track['age'] < self.max_age]

        return tracked_objects

    def get_tracks(self):
        """Return all currently tracked objects."""
        return self.tracks

