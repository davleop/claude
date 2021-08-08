import numpy as np
import claude_low_level_library as low_level
import claude_top_level_library as top_level

def grid_lat(mul, xx, yy, rad):
    return (mul * np.arccos(((xx**2 + yy**2)**0.5)/rad)*180.0/np.pi).flatten()

def grid_lon(xx, yy):
    return (180.0 - np.arctan2(yy,xx)*180.0/np.pi).flatten()

def cos_mul_sin(rad, lat, lon, i, j):
    return rad * np.cos(lat[i]*np.pi/180.0) * np.sin(lon[j]*np.pi/180.0)

def cos_mul_cos(rad, lat, lon, i, j):
    return rad * np.cos(lat[i]*np.pi/180.0) * np.cos(lon[j]*np.pi/180.0)

