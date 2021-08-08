"""CLimate Analysis using Digital Estimations Backend (CLAuDE Backend)

TODO(): add proper documentation here
"""

# preinstalled libs
import sys
import time
import pickle 
import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy

# Local libs
import claude_low_level_library as low_level
import claude_top_level_library as top_level

from util import *
from Settings import Settings

class ClaudeBackend:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

        for item in dir(self.settings): # this assumes 
            if not item.startswith("__"):
                try:
                    exec(f"self.{item} = self.settings.{item}")
                except: # probably catch the exact exception, but eh
                    exec("self.{} = self.settings.{}".format(item, item))

        self._setup()
        self._init()
        self._setup_grids()
        self._plot()

    def load(self, save_file: str="save_file.p") -> None:
        tup = pickle.load(open(save_file,"rb"))
        self.potential_temperature = tup[0]
        self.temperature_world = tup[1]
        self.u, self.v, self.w = tup[2], tup[3], tup[4]
        self.x_dot_N, self.y_dot_N = tup[5], tup[6]
        self.x_dot_S, self.y_dot_S = tup[7], tup[8]
        self.t = tup[9]
        self.albedo = tup[10]
        self.tracer = tup[11]

        self.tracer = np.zeros_like(potential_temperature)
        self.last_plot = self.t-0.1
        self.last_save = self.t-0.1

    def _setup(self) -> None:
        self.temperature_world += 200
        self.potential_temperature = np.zeros((self.nlat,self.nlon,self.nlevels))
        self.u = np.zeros_like(self.potential_temperature)
        self.v = np.zeros_like(self.potential_temperature)
        self.w = np.zeros_like(self.potential_temperature)
        self.atmosp_addition = np.zeros_like(self.potential_temperature)

        # read temperature and density in from standard atmosphere
        standard_temp     = []
        standard_pressure = []
        with open(self.standard_atmosphere, "r") as f:
            for x in f:
                h, t, r, p = x.split()
                standard_temp.append(float(t))
                standard_pressure.append(float(p))

        # density_profile = np.interp(x=heights/1E3,xp=standard_height,fp=standard_density)
        temp_profile = np.interp(x  = self.pressure_levels[::-1],
                                 xp = standard_pressure   [::-1],
                                 fp = standard_temp       [::-1]) [::-1]
        for k in range(self.nlevels):
            self.potential_temperature[:,:,k] = temp_profile[k]

        self.potential_temperature = low_level.t_to_theta(self.potential_temperature,self.pressure_levels)
        self.geopotential = np.zeros_like(self.potential_temperature)
        self.tracer = np.zeros_like(self.potential_temperature)

    def _init(self) -> None:
        self.sigma = np.zeros_like(self.pressure_levels)
        self.kappa = 287/1000
        self.heat_capacity_earth = np.zeros_like(self.temperature_world) + 1E6
        albedo_variance = 0.001
        specific_gas = 287
        thermal_diffusivity_roc = 1.5E-6
        angular_speed = 2*np.pi/self.day

        for i in range(len(self.sigma)):
            self.sigma[i] = 1E3*(self.pressure_levels[i]/self.pressure_levels[0])**self.kappa

        # heat_capacity_earth[15:36,30:60] = 1E7
        # heat_capacity_earth[30:40,80:90] = 1E7

        self.albedo = np.random.uniform(-albedo_variance,albedo_variance, (self.nlat, self.nlon)) + 0.2

        # define planet size and various geometric constants
        circumference = 2*np.pi*self.planet_radius
        circle = np.pi*self.planet_radius**2
        sphere = 4*np.pi*self.planet_radius**2

        # define how far apart the gridpoints are: note that we use central difference derivatives,
        # and so these distances are actually twice the distance between gridboxes
        self.dy = circumference/self.nlat
        self.dx = np.zeros(self.nlat)
        self.coriolis = np.zeros(self.nlat)   # also define the coriolis parameter here
        for i in range(self.nlat):
            self.dx[i] = self.dy*np.cos(self.lat[i]*np.pi/180)
            self.coriolis[i] = angular_speed*np.sin(self.lat[i]*np.pi/180)

        self.sponge_index = np.where(self.pressure_levels < self.sponge_layer*100)[0][0]

    def _setup_grids(self) -> None:
        grid_pad = 2

        # N/S indices
        self.pole_low_index_S  = np.where(self.lat >  self.pole_lower_latitude_limit) [0][0]
        self.pole_high_index_S = np.where(self.lat >  self.pole_higher_latitude_limit)[0][0]
        self.pole_low_index_N  = np.where(self.lat < -self.pole_lower_latitude_limit) [0][-1]
        self.pole_high_index_N = np.where(self.lat < -self.pole_higher_latitude_limit)[0][-1]

        self.indices = (self.pole_low_index_N,self.pole_high_index_N,self.pole_low_index_S,self.pole_high_index_S)

        # Initialise Grid
        self.polar_grid_resolution = self.dx[self.pole_low_index_S]
        size_of_grid = self.planet_radius*np.cos(self.lat[self.pole_low_index_S+grid_pad]*np.pi/180.0)

        ### Grid X/Y Values N/S
        self.grid_x_values_S = np.arange(-size_of_grid,size_of_grid,self.polar_grid_resolution)
        self.grid_y_values_S = deepcopy(self.grid_x_values_S)
        self.grid_x_values_N = deepcopy(self.grid_x_values_S)
        self.grid_y_values_N = deepcopy(self.grid_x_values_S)
        
        grid_xx_S,grid_yy_S = np.meshgrid(self.grid_x_values_S,self.grid_y_values_S)
        grid_xx_N,grid_yy_N =   np.meshgrid(self.grid_x_values_N,self.grid_y_values_N)

        self.grid_side_length = len(self.grid_x_values_S)

        ### N/S Lat/Lon Coords
        self.grid_lat_coords_S = grid_lat(-1, grid_xx_S, grid_yy_S, self.planet_radius)
        self.grid_lon_coords_S = grid_lon(grid_xx_S, grid_yy_S)
        self.grid_lat_coords_N = grid_lat(1, grid_xx_N, grid_yy_N, self.planet_radius)
        self.grid_lon_coords_N = grid_lon(grid_xx_N, grid_yy_N)

        self.polar_x_coords_S = []
        self.polar_y_coords_S = []
        self.polar_x_coords_N = []
        self.polar_y_coords_N = []

        for i in range(self.pole_low_index_S):
            for j in range(self.nlon):
                self.polar_x_coords_S.append( cos_mul_sin(self.planet_radius, self.lat, self.lon, i, j) )
                self.polar_y_coords_S.append( cos_mul_cos(-self.planet_radius, self.lat, self.lon, i, j) )

        for i in np.arange(self.pole_low_index_N,self.nlat):
            for j in range(self.nlon):
                self.polar_x_coords_N.append( cos_mul_sin(self.planet_radius, self.lat, self.lon, i, j) )
                self.polar_y_coords_N.append( cos_mul_cos(-self.planet_radius, self.lat, self.lon, i, j) )

        self.grids = (grid_xx_N.shape[0],grid_xx_S.shape[0])

        # create Coriolis data on north and south planes
        self.data = np.zeros((self.nlat-self.pole_low_index_N+grid_pad,self.nlon))
        for i in np.arange(self.pole_low_index_N-grid_pad,self.nlat):
            self.data[i-self.pole_low_index_N,:] = self.coriolis[i]
        self.coriolis_plane_N = low_level.beam_me_up_2D(self.lat[(self.pole_low_index_N-grid_pad):],self.lon,self.data,self.grids[0],self.grid_lat_coords_N,self.grid_lon_coords_N)
        self.data = np.zeros((self.pole_low_index_S+grid_pad,self.nlon))
        for i in range(self.pole_low_index_S+grid_pad):
            self.data[i,:] = self.coriolis[i]
        self.coriolis_plane_S = low_level.beam_me_up_2D(self.lat[:(self.pole_low_index_S+grid_pad)],self.lon,self.data,self.grids[1],self.grid_lat_coords_S,self.grid_lon_coords_S)

        self.x_dot_N = np.zeros((self.grids[0],self.grids[0],self.nlevels))
        self.y_dot_N = np.zeros((self.grids[0],self.grids[0],self.nlevels))
        self.x_dot_S = np.zeros((self.grids[1],self.grids[1],self.nlevels))
        self.y_dot_S = np.zeros((self.grids[1],self.grids[1],self.nlevels))

        N_grid = self.grid_lat_coords_N,self.grid_lon_coords_N
        N_vals = self.grid_x_values_N,self.grid_y_values_N
        N_pols = self.polar_x_coords_N,self.polar_y_coords_N

        S_grid = self.grid_lat_coords_S,self.grid_lon_coords_S
        S_vals = self.grid_x_values_S,self.grid_y_values_S
        S_pols = self.polar_x_coords_S,self.polar_y_coords_S

        self.coords = N_grid + N_vals + N_pols + S_grid + S_vals + S_pols

    def _plot(self) -> None:
        if not self.diagnostic:
            # set up plot
            self.f, self.ax = plt.subplots(2,figsize=(9,9))
            self.f.canvas.set_window_title('CLAuDE')
            self.ax[0].contourf(self.lon_plot, self.lat_plot, self.temperature_world, cmap='seismic')
            
            self.ax[0].streamplot(self.lon_plot,
                                  self.lat_plot, 
                                  self.u[:,:,0], 
                                  self.v[:,:,0], 
                                  color='white',
                                  density=1)

            test = self.ax[1].contourf(self.heights_plot, 
                                       self.lat_z_plot, 
                                       np.transpose(np.mean(
                                            low_level.theta_to_t(self.potential_temperature,self.pressure_levels),
                                            axis=1))[:self.top,:],
                                       cmap='seismic',
                                       levels=15)

            self.ax[1].contour(self.heights_plot,
                               self.lat_z_plot, 
                               np.transpose(np.mean(self.u,axis=1))[:self.top,:], 
                               colors='white',
                               levels=20,
                               linewidths=1,
                               alpha=0.8)

            self.ax[1].quiver(self.heights_plot, 
                              self.lat_z_plot, 
                              np.transpose(np.mean(self.v,axis=1))[:self.top,:],
                              np.transpose(np.mean(100*self.w,axis=1))[:self.top,:],
                              color='black')

            plt.subplots_adjust(left=0.1, right=0.75)
            self.ax[0].set_title('Surface temperature')
            self.ax[0].set_xlim(self.lon.min(),self.lon.max())
            self.ax[1].set_title('Atmosphere temperature')
            self.ax[1].set_xlim(self.lat.min(),self.lat.max())
            self.ax[1].set_ylim((self.pressure_levels.max()/100,self.pressure_levels[:self.top].min()/100))
            self.ax[1].set_yscale('log')
            self.ax[1].set_ylabel('Pressure (hPa)')
            self.ax[1].set_xlabel('Latitude')
            self.cbar_ax = self.f.add_axes([0.85, 0.15, 0.05, 0.7])
            self.f.colorbar(test, cax=self.cbar_ax)
            self.cbar_ax.set_title('Temperature (K)')
            self.f.suptitle( 'Time ' + str(round(self.t/self.day,2)) + ' days' )

        else:
            # set up plot
            self.f, self.ax = plt.subplots(2,2,figsize=(9,9))
            self.f.canvas.set_window_title('CLAuDE')
            self.ax[0,0].contourf(self.heights_plot, 
                                  self.lat_z_plot, 
                                  np.transpose(np.mean(self.u,axis=1))[:self.top,:], 
                                  cmap='seismic')
            self.ax[0,0].set_title('u')
            self.ax[0,1].contourf(self.heights_plot, 
                                  self.lat_z_plot, 
                                  np.transpose(np.mean(self.v,axis=1))[:self.top,:], 
                                  cmap='seismic')
            self.ax[0,1].set_title('v')
            self.ax[1,0].contourf(self.heights_plot, 
                                  self.lat_z_plot, 
                                  np.transpose(np.mean(self.w,axis=1))[:self.top,:], 
                                  cmap='seismic')
            self.ax[1,0].set_title('w')
            self.ax[1,1].contourf(self.heights_plot, 
                                  self.lat_z_plot, 
                                  np.transpose(np.mean(self.atmosp_addition,axis=1))[:self.top,:], 
                                  cmap='seismic')
            self.ax[1,1].set_title('atmosp_addition')
            for axis in ax.ravel():
                axis.set_ylim((self.pressure_levels.max()/100,self.pressure_levels[:self.top].min()/100))
                axis.set_yscale('log')
            self.f.suptitle( 'Time ' + str(round(self.t/self.day,2)) + ' days' )

        if self.level_plots:
            self.level_divisions = int(np.floor(self.nlevels/self.nplots))
            self.level_plots_levels = range(self.nlevels)[::self.level_divisions][::-1]

            self.g, self.bx = plt.subplots(self.nplots,figsize=(9,8),sharex=True)
            self.g.canvas.set_window_title('CLAuDE pressure levels')
            for k, z in zip(range(self.nplots), self.level_plots_levels): 
                z += 1
                self.bx[k].contourf(self.lon_plot, 
                                    self.lat_plot, 
                                    self.potential_temperature[:,:,z], 
                                    cmap='seismic')
                self.bx[k].set_title(str(self.pressure_levels[z]/100)+' hPa')
                self.bx[k].set_ylabel('Latitude')
            self.bx[-1].set_xlabel('Longitude')

        plt.ion()
        plt.show()
        plt.pause(2)
        
        if not self.diagnostic:
            self.ax[0].cla()
            self.ax[1].cla()
            if self.level_plots:
                for k in range(self.nplots):
                    self.bx[k].cla()     
        else:
            self.ax[0,0].cla()
            self.ax[0,1].cla() 
            self.ax[1,0].cla()
            self.ax[1,1].cla()

        if self.above:
            self.g, self.gx = plt.subplots(1,3, figsize=(15,5))
            plt.ion()
            plt.show()

    def _plotting_routine(self) -> None:
        quiver_padding = int(12/self.resolution)

        if self.plot:
            if self.verbose: before_plot = time.time()
            # update plot
            if not self.diagnostic:
                
                field = np.copy(self.w)[:,:,self.sample_level]
                test = self.ax[0].contourf(self.lon_plot, self.lat_plot, field, cmap='seismic',levels=15)
                self.ax[0].contour(self.lon_plot, 
                                   self.lat_plot, 
                                   self.tracer[:,:,self.sample_level], 
                                   alpha=0.5, 
                                   antialiased=True, 
                                   levels=np.arange(0.01,1.01,0.01))
                if self.velocity: 
                    self.ax[0].quiver(self.lon_plot[::quiver_padding,::quiver_padding], 
                                      self.lat_plot[::quiver_padding,::quiver_padding], 
                                      self.u[::quiver_padding,::quiver_padding,self.sample_level], 
                                      self.v[::quiver_padding,::quiver_padding,self.sample_level], 
                                      color='white')
                self.ax[0].set_xlim((self.lon.min(),self.lon.max()))
                self.ax[0].set_ylim((self.lat.min(),self.lat.max()))
                self.ax[0].set_ylabel('Latitude')
                self.ax[0].axhline(y=0,color='black',alpha=0.3)
                self.ax[0].set_xlabel('Longitude')

                test = self.ax[1].contourf(self.heights_plot, 
                                           self.lat_z_plot, 
                                           np.transpose(np.mean(low_level.theta_to_t(self.potential_temperature,
                                                                                     self.pressure_levels),axis=1))[:self.top,:], 
                                           cmap='seismic',levels=15)
                self.ax[1].contour(self.heights_plot, 
                                   self.lat_z_plot, 
                                   np.transpose(np.mean(self.tracer,axis=1))[:self.top,:], 
                                   alpha=0.5, 
                                   antialiased=True, 
                                   levels=np.arange(0.001,1.01,0.01))

                if self.velocity:
                    self.ax[1].contour(self.heights_plot,
                                       self.lat_z_plot, 
                                       np.transpose(np.mean(self.u,axis=1))[:self.top,:], 
                                       colors='white',
                                       levels=20,
                                       linewidths=1,
                                       alpha=0.8)
                    self.ax[1].quiver(self.heights_plot, 
                                      self.lat_z_plot, 
                                      np.transpose(np.mean(self.v,axis=1))[:self.top,:],
                                      np.transpose(np.mean(-10*self.w,axis=1))[:self.top,:],
                                      color='black')
                self.ax[1].set_title('$\it{Atmospheric} \quad \it{temperature}$')
                self.ax[1].set_xlim((-90,90))
                self.ax[1].set_ylim((self.pressure_levels.max()/100,self.pressure_levels[:self.top].min()/100))
                self.ax[1].set_ylabel('Pressure (hPa)')
                self.ax[1].set_xlabel('Latitude')
                self.ax[1].set_yscale('log')
                self.f.colorbar(test, cax=self.cbar_ax)
                self.cbar_ax.set_title('Temperature (K)')
                self.f.suptitle( 'Time ' + str(round(self.t/self.day,2)) + ' days' )
                    
            else:
                self.ax[0,0].contourf(self.heights_plot, 
                                      self.lat_z_plot, 
                                      np.transpose(np.mean(self.u,axis=1))[:self.top,:], 
                                      cmap='seismic')
                self.ax[0,0].set_title('u')

                self.ax[0,1].contourf(self.heights_plot, 
                                      self.lat_z_plot, 
                                      np.transpose(np.mean(self.v,axis=1))[:self.top,:], 
                                      cmap='seismic')
                self.ax[0,1].set_title('v')
                
                self.ax[1,0].contourf(self.heights_plot, 
                                      self.lat_z_plot, 
                                      np.transpose(np.mean(self.w,axis=1))[:self.top,:], 
                                      cmap='seismic')
                self.ax[1,0].set_title('w')
                
                self.ax[1,1].contourf(self.heights_plot, 
                                      self.lat_z_plot, 
                                      np.transpose(np.mean(self.atmosp_addition,axis=1))[:self.top,:], 
                                      cmap='seismic')
                self.ax[1,1].set_title('atmosp_addition')
                
                for axis in self.ax.ravel():
                    axis.set_ylim((self.pressure_levels.max()/100,self.pressure_levels[:self.top].min()/100))
                    axis.set_yscale('log')
                self.f.suptitle( 'Time ' + str(round(self.t/self.day,2)) + ' days' )

            if self.level_plots:
                for k, z in zip(range(self.nplots), self.level_plots_levels): 
                    z += 1
                    self.bx[k].contourf(self.lon_plot, 
                                        self.lat_plot, 
                                        self.potential_temperature[:,:,z], 
                                        cmap='seismic',
                                        levels=15)
                    self.bx[k].quiver(self.lon_plot[::quiver_padding,::quiver_padding], 
                                      self.lat_plot[::quiver_padding,::quiver_padding], 
                                      self.u[::quiver_padding,::quiver_padding,z], 
                                      self.v[::quiver_padding,::quiver_padding,z], 
                                      color='white')
                    self.bx[k].set_title(str(round(p.pressure_levels[z]/100))+' hPa')
                    self.bx[k].set_ylabel('Latitude')
                    self.bx[k].set_xlim((self.lon.min(),self.lon.max()))
                    self.bx[k].set_ylim((self.lat.min(),self.lat.max()))               
                self.bx[-1].set_xlabel('Longitude')

        if self.above and self.velocity:
            self.gx[0].set_title('Original data')
            self.gx[1].set_title('Polar plane')
            self.gx[2].set_title('Reprojected data')
            self.g.suptitle( 'Time ' + str(round(self.t/self.day,2)) + ' days' )

            if self.pole == 's':
                self.gx[0].set_title('temperature')
                self.gx[0].contourf(self.lon,
                                    self.lat[:self.pole_low_index_S],
                                    self.potential_temperature[:self.pole_low_index_S,:,self.above_level])
                
                self.gx[1].set_title('polar_plane_advect')
                self.polar_temps = low_level.beam_me_up(self.lat[:self.pole_low_index_S],
                                                        self.lon,
                                                        self.potential_temperature[:self.pole_low_index_S,:,:],
                                                        self.grids[1],
                                                        self.grid_lat_coords_S,
                                                        self.grid_lon_coords_S)
                output = low_level.beam_me_up_2D(self.lat[:self.pole_low_index_S],
                                                 self.lon,
                                                 self.w[:self.pole_low_index_S,:,self.above_level],
                                                 self.grids[1],
                                                 self.grid_lat_coords_S,
                                                 self.grid_lon_coords_S)

                self.gx[1].contourf(self.grid_x_values_S/1E3,self.grid_y_values_S/1E3,output)
                self.gx[1].contour(self.grid_x_values_S/1E3,
                                   self.grid_y_values_S/1E3,
                                   self.polar_temps[:,:,self.above_level],
                                   colors='white',
                                   levels=20,
                                   linewidths=1,
                                   alpha=0.8)
                self.gx[1].quiver(self.grid_x_values_S/1E3,
                                  self.grid_y_values_S/1E3,
                                  self.x_dot_S[:,:,self.above_level],
                                  self.y_dot_S[:,:,self.above_level])
                
                self.gx[1].add_patch(plt.Circle((0,0),
                                     p.planet_radius*np.cos(self.lat[self.pole_low_index_S]*np.pi/180.0)/1E3,
                                     color='r',
                                     fill=False))
                self.gx[1].add_patch(plt.Circle((0,0),
                                     p.planet_radius*np.cos(self.lat[self.pole_high_index_S]*np.pi/180.0)/1E3,
                                     color='r',
                                     fill=False))

                self.gx[2].set_title('south_addition_smoothed')
                self.gx[2].contourf(self.lon,
                                    self.lat[:self.pole_low_index_S],
                                    self.u[:self.pole_low_index_S,:,self.above_level])
                self.gx[2].quiver(self.lon[::5],
                                  self.lat[:self.pole_low_index_S],
                                  self.u[:self.pole_low_index_S,::5,self.above_level],
                                  self.v[:self.pole_low_index_S,::5,self.above_level])
            else:
                self.gx[0].set_title('temperature')
                self.gx[0].contourf(self.lon,
                                    self.lat[self.pole_low_index_N:],
                                    self.potential_temperature[self.pole_low_index_N:,:,self.above_level])
                
                self.gx[1].set_title('polar_plane_advect')
                self.polar_temps = low_level.beam_me_up(self.lat[self.pole_low_index_N:],
                                                        self.lon,
                                                        np.flip(self.potential_temperature[self.pole_low_index_N:,:,:],axis=1),
                                                        self.grids[0],
                                                        self.grid_lat_coords_N,
                                                        self.grid_lon_coords_N)
                output = low_level.beam_me_up_2D(self.lat[self.pole_low_index_N:],
                                                 self.lon,
                                                 self.atmosp_addition[self.pole_low_index_N:,:,self.above_level],
                                                 self.grids[0],
                                                 self.grid_lat_coords_N,
                                                 self.grid_lon_coords_N)
                output = low_level.beam_me_up_2D(self.lat[self.pole_low_index_N:],
                                                 self.lon,
                                                 self.w[self.pole_low_index_N:,:,self.above_level],
                                                 self.grids[0],
                                                 self.grid_lat_coords_N,
                                                 self.grid_lon_coords_N)

                self.gx[1].contourf(self.grid_x_values_N/1E3,self.grid_y_values_N/1E3,output)
                self.gx[1].contour(self.grid_x_values_N/1E3,
                                   self.grid_y_values_N/1E3,
                                   self.polar_temps[:,:,self.above_level],
                                   colors='white',
                                   levels=20,
                                   linewidths=1,
                                   alpha=0.8)
                self.gx[1].quiver(self.grid_x_values_N/1E3,
                                  self.grid_y_values_N/1E3,
                                  self.x_dot_N[:,:,self.above_level],
                                  self.y_dot_N[:,:,self.above_level])
                
                self.gx[1].add_patch(plt.Circle((0,0),
                                     self.planet_radius*np.cos(self.lat[self.pole_low_index_N]*np.pi/180.0)/1E3,
                                     color='r',
                                     fill=False))
                self.gx[1].add_patch(plt.Circle((0,0),
                                     self.planet_radius*np.cos(self.lat[self.pole_high_index_N]*np.pi/180.0)/1E3,
                                     color='r',
                                     fill=False))
        
                self.gx[2].set_title('south_addition_smoothed')
                self.gx[2].contourf(self.lon,
                                    self.lat[self.pole_low_index_N:],
                                    self.u[self.pole_low_index_N:,:,self.above_level])
                self.gx[2].quiver(self.lon[::5],
                                  self.lat[self.pole_low_index_N:],
                                  self.u[self.pole_low_index_N:,::5,self.above_level],
                                  self.v[self.pole_low_index_N:,::5,self.above_level])
            
        # clear plots
        if self.plot or self.above:   plt.pause(0.001)
        if self.plot:
            if not self.diagnostic:
                self.ax[0].cla()
                self.ax[1].cla()
                self.cbar_ax.cla()
                        
            else:
                self.ax[0,0].cla()
                self.ax[0,1].cla()
                self.ax[1,0].cla()
                self.ax[1,1].cla()
            if self.level_plots:
                for k in range(self.nplots):
                    self.bx[k].cla() 
            if self.verbose:     
                self.time_taken = float(round(time.time() - before_plot,3))
                print('Plotting: ',str(self.time_taken),'s') 
        if self.above:
            self.gx[0].cla()
            self.gx[1].cla()
            self.gx[2].cla()