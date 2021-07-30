"""Main

driver for running CLAuDE
"""

from Structures import PlanetSettings
from Structures import SmoothingSettings
from Structures import SaveSettings
from Structures import PlotSettings
from Structures import CoordinateSettings

from Claude import Claude

def main():
    # DEFINE SETTINGS HERE
    planet_settings = PlanetSettings()
    smoothing_settings = SmoothingSettings()
    save_settings = SaveSettings()
    plot_settings = PlotSettings()
    coordinate_settings = CoordinateSettings()

    claude = Claude(planet_settings, smoothing_settings, save_settings, plot_settings, coordinate_settings)

    # claude.setup()
    # claude.init()
    # claude.setup_grids()
    # claude.plot()
    # claude.run()

if __name__ == '__main__':
    main()