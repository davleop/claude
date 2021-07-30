"""CLimate Analysis using Digital Estimations (CLAuDE)

TODO(): add proper documentation here
"""

# pip installed libs
import sys
import time
import pickle 
import numpy as np 
import matplotlib.pyplot as plt

# Local libs
import claude_low_level_library as low_level
import claude_top_level_library as top_level

from Structures import SaveSettings
from Structures import PlotSettings
from Structures import PlanetSettings
from Structures import SmoothingSettings
from Structures import CoordinateSettings

class Claude:
    def __init__(self, planet: PlanetSettings,
                       smoothing: SmoothingSettings,
                       saving: SaveSettings,
                       plotting: PlotSettings,
                       coordinates: CoordinateSettings) -> None:
        self.planet = planet
        self.smoothing = smoothing
        self.saving = saving
        self.plotting = plotting
        self.coordinates = coordinates