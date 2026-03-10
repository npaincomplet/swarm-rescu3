import numpy as np
from typing import List
from swarm_rescue.solutions.utils.dataclasses_config import ExplorationTrackerParams

class ExplorationTracker:
    def __init__(self):
        self.wounded_sighting_positions: List[np.array] = []
        self.exploration_tracker_params = ExplorationTrackerParams
    
    def _wounded_sighting_is_new(self, sighting_position):
        """
        A wounded sighting is considered new if it's far enough from every other already registered sightings.
        This might still generate duplicate sightings for the same wounded because of noise and because wounded can move.
        What matters is knowing where could wounded be without having an infinitely growing list of sightings.
        """
        for existing_sighting in self.wounded_sighting_positions:
            if np.linalg.norm(np.array(sighting_position) - np.array(existing_sighting)) < self.exploration_tracker_params.SAME_WOUNDED_RADIUS:
                return False
        return True

    def merge_wounded_sighting(self, received_sighting_positions):
        """Merge received wounded sightings with existing ones, keeping only unique sightings"""
        for received_position in received_sighting_positions:
            if self._wounded_sighting_is_new(received_position):
                self.wounded_sighting_positions.append(received_position)