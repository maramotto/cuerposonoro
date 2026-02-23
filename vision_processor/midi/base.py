"""
Base interface for all MIDI sender implementations.

Any sender that plugs into the CuerpoSonoro pipeline must implement
these two methods. The rest of the pipeline (main.py, Config) only
speaks to this interface — it doesn't know which implementation is
running underneath.
"""

from abc import ABC, abstractmethod


class BaseMidiSender(ABC):
    """Common interface for MIDI output strategies."""

    @abstractmethod
    def update(self, features: dict):
        """
        Called once per frame with the latest feature values.

        Args:
            features: Dict produced by FeatureExtractor.calculate().
                      Keys include feetCenterX, hipTilt, rightHandY,
                      rightHandJerk, energy, headTilt, etc.
        """

    @abstractmethod
    def close(self):
        """
        Send All Notes Off and close the MIDI port cleanly.
        Called once on program exit.
        """
