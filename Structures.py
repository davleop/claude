"""Structures

TODO(): add proper documentation
"""

import numpy as np 

class PlanetSettings:
    def __init__(self):
        self.day           = 60*60*24
        self.resolution    = 3
        self.planet_radius = 6.4E6
        self.insolation    = 1370
        self.gravity       = 9.81
        self.axial_tilt    = 23.5
        self.year          = 365*self.day

        self.pressure_levels  = np.array([1000,950,900,800,700,600,500,400,350,300,250,200,150,100,75,50,25,10,5,2,1])
        self.pressure_levels *= 100
        self.nlevels          = len(self.pressure_levels)

        self.dt_spinup     = 60*17.2
        self.dt_main       = 60*7.2
        self.spinup_length = 0*self.day

class SmoothingSettings:
    def __init__(self):
        self.smoothing = False
        self.smoothing_parameter_t   = 1.0
        self.smoothing_parameter_u   = 0.9
        self.smoothing_parameter_v   = 0.9
        self.smoothing_parameter_w   = 0.3
        self.smoothing_parameter_add = 0.3

class SaveSettings:
    def __init__(self):
        self.save = False
        self.load = False
        self.save_frequency = 100

class PlotSettings(PlanetSettings):
    def __init__(self):
        super().__init__()
        self.plot_frequency = 10
        self.above          = False
        self.pole           = 's'
        self.above_level    = 17

        self.plot        = True
        self.diagnostic  = False
        self.level_plots = False
        self.nplots      = 3
        self.top         = 17


class CoordinateSettings(PlotSettings):
    def __init__(self):
        super().__init__()
        self.pole_lower_latitude_limit  = -75
        self.pole_higher_latitude_limit = -85
        self.sponge_layer               = 10

        self.lat = np.arange(-90,91,self.resolution)
        self.lon = np.arange(0,360,self.resolution)
        
        self.nlat = len(self.lat)
        self.nlon = len(self.lon)
        
        self.lon_plot, self.lat_plot       = np.meshgrid(self.lon, self.lat)
        self.heights_plot, self.lat_z_plot = np.meshgrid(self.lat,self.pressure_levels[:self.top]/100)
        self.temperature_world             = np.zeros((self.nlat,self.nlon))

