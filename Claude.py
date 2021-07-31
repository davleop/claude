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

        self.standard_atmosphere = "standard_atmosphere.txt"
        self.verbose = False

        self.potential_temperature = None
        self.geopotential = None

        self.t = 0.0
        self.sample_level = 15
        self.last_plot = self.t-0.1
        self.last_save = self.t-0.1

    def setup(self):
        # tmp reference
        c = self.coordinates
        p = self.planet
        
        c.temperature_world += 200
        self.potential_temperature = np.zeros((c.nlat,c.nlon,p.nlevels))
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
        temp_profile = np.interp(x=p.pressure_levels[::-1],xp=standard_pressure[::-1],fp=standard_temp[::-1])[::-1]
        for k in range(p.nlevels):
            self.potential_temperature[:,:,k] = temp_profile[k]

        self.potential_temperature = low_level.t_to_theta(self.potential_temperature,p.pressure_levels)
        self.geopotential = np.zeros_like(self.potential_temperature)
        self.tracer = np.zeros_like(potential_temperature)

    def init(self):
        # tmp reference
        p = self.planet
        c = self.coordinates

        sigma = np.zeros_like(p.pressure_levels)
        kappa = 287/1000
        for i in range(len(sigma)):
            sigma[i] = 1E3*(p.pressure_levels[i]/p.pressure_levels[0])**kappa

        self.heat_capacity_earth = np.zeros_like(c.temperature_world) + 1E6

        # heat_capacity_earth[15:36,30:60] = 1E7
        # heat_capacity_earth[30:40,80:90] = 1E7

        albedo_variance = 0.001
        self.albedo = np.random.uniform(-albedo_variance,albedo_variance, (nlat, nlon)) + 0.2
        # albedo = np.zeros((nlat, nlon)) + 0.2

        specific_gas = 287
        thermal_diffusivity_roc = 1.5E-6

        # define planet size and various geometric constants
        circumference = 2*np.pi*p.planet_radius
        circle = np.pi*p.planet_radius**2
        sphere = 4*np.pi*p.planet_radius**2

        # define how far apart the gridpoints are: note that we use central difference derivatives,
        # and so these distances are actually twice the distance between gridboxes
        dy = circumference/c.nlat
        dx = np.zeros(c.nlat)
        self.coriolis = np.zeros(c.nlat)   # also define the coriolis parameter here
        angular_speed = 2*np.pi/p.day
        for i in range(nlat):
            dx[i] = dy*np.cos(lat[i]*np.pi/180)
            self.coriolis[i] = angular_speed*np.sin(lat[i]*np.pi/180)

        self.sponge_index = np.where(p.pressure_levels < c.sponge_layer*100)[0][0]

    def setup_grids(self):
        # tmp references
        c = self.coordinates
        p = self.planet

        grid_pad = 2
    
        self.pole_low_index_S = np.where(lat > c.pole_lower_latitude_limit)[0][0]
        self.self.pole_high_index_S = np.where(lat > c.pole_higher_latitude_limit)[0][0]

        # initialise grid
        self.polar_grid_resolution = dx[self.pole_low_index_S]
        size_of_grid = p.planet_radius*np.cos(lat[self.pole_low_index_S+grid_pad]*np.pi/180.0)

        ### south pole ###
        self.grid_x_values_S = np.arange(-size_of_grid,size_of_grid,polar_grid_resolution)
        self.grid_y_values_S = np.arange(-size_of_grid,size_of_grid,polar_grid_resolution)
        grid_xx_S,grid_yy_S = np.meshgrid(self.grid_x_values_S,self.grid_y_values_S)

        self.grid_side_length = len(self.grid_x_values_S)

        self.grid_lat_coords_S = (-np.arccos(((grid_xx_S**2 + grid_yy_S**2)**0.5)/p.planet_radius)*180.0/np.pi).flatten()
        self.grid_lon_coords_S = (180.0 - np.arctan2(grid_yy_S,grid_xx_S)*180.0/np.pi).flatten()

        self.polar_x_coords_S = []
        self.polar_y_coords_S = []
        for i in range(self.pole_low_index_S):
            for j in range(nlon):
                self.polar_x_coords_S.append( p.planet_radius*np.cos(c.lat[i]*np.pi/180.0)*np.sin(c.lon[j]*np.pi/180.0) )
                self.polar_y_coords_S.append(-p.planet_radius*np.cos(c.lat[i]*np.pi/180.0)*np.cos(c.lon[j]*np.pi/180.0) )

        ### north pole ###
        
        self.pole_low_index_N    =   np.where(lat < -pole_lower_latitude_limit)[0][-1]
        self.pole_high_index_N   =   np.where(lat < -pole_higher_latitude_limit)[0][-1]

        self.grid_x_values_N     =   np.arange(-size_of_grid,size_of_grid,polar_grid_resolution)
        self.grid_y_values_N     =   np.arange(-size_of_grid,size_of_grid,polar_grid_resolution)
        grid_xx_N,grid_yy_N =   np.meshgrid(self.grid_x_values_N,self.grid_y_values_N)

        self.grid_lat_coords_N   =   (np.arccos((grid_xx_N**2 + grid_yy_N**2)**0.5/p.planet_radius)*180.0/np.pi).flatten()
        self.grid_lon_coords_N   =   (180.0 - np.arctan2(grid_yy_N,grid_xx_N)*180.0/np.pi).flatten()

        self.polar_x_coords_N    =   []
        self.polar_y_coords_N    =   []
        for i in np.arange(self.pole_low_index_N,nlat):
            for j in range(nlon):
                self.polar_x_coords_N.append( p.planet_radius*np.cos(c.lat[i]*np.pi/180.0)*np.sin(c.lon[j]*np.pi/180.0) )
                self.polar_y_coords_N.append(-p.planet_radius*np.cos(c.lat[i]*np.pi/180.0)*np.cos(c.lon[j]*np.pi/180.0) )

        self.indices = (self.pole_low_index_N,self.pole_high_index_N,self.pole_low_index_S,self.pole_high_index_S)
        self.grids   = (grid_xx_N.shape[0],grid_xx_S.shape[0])

        # create Coriolis data on north and south planes
        self.data = np.zeros((nlat-self.pole_low_index_N+grid_pad,c.nlon))
        for i in np.arange(self.pole_low_index_N-grid_pad,c.nlat):
            self.data[i-self.pole_low_index_N,:] = self.coriolis[i]
        self.coriolis_plane_N = low_level.beam_me_up_2D(c.lat[(self.pole_low_index_N-grid_pad):],c.lon,self.data,self.grids[0],self.grid_lat_coords_N,self.grid_lon_coords_N)
        self.data = np.zeros((self.pole_low_index_S+grid_pad,c.nlon))
        for i in range(self.pole_low_index_S+grid_pad):
            self.data[i,:] = self.coriolis[i]
        self.coriolis_plane_S = low_level.beam_me_up_2D(c.lat[:(self.pole_low_index_S+grid_pad)],c.lon,self.data,self.grids[1],self.grid_lat_coords_S,self.grid_lon_coords_S)

        self.x_dot_N = np.zeros((self.grids[0],self.grids[0],p.nlevels))
        self.y_dot_N = np.zeros((self.grids[0],self.grids[0],p.nlevels))
        self.x_dot_S = np.zeros((self.grids[1],self.grids[1],p.nlevels))
        self.y_dot_S = np.zeros((self.grids[1],self.grids[1],p.nlevels))

        self.coords  = self.grid_lat_coords_N,self.grid_lon_coords_N,self.grid_x_values_N,self.grid_y_values_N,self.polar_x_coords_N,self.polar_y_coords_N,self.grid_lat_coords_S,self.grid_lon_coords_S,self.grid_x_values_S,self.grid_y_values_S,self.polar_x_coords_S,self.polar_y_coords_S  

    def load(self, save_file="save_file.p"):
        self.potential_temperature,self.temperature_world,self.u,self.v,self.w,self.x_dot_N,self.y_dot_N,self.x_dot_S,self.y_dot_S,self.t,self.albedo,self.tracer = pickle.load(open(save_file,"rb"))
        self.tracer = np.zeros_like(potential_temperature)

        self.last_plot = t-0.1
        self.last_save = t-0.1

    def plot(self):
        # tmp references 
        pl = self.plotting
        c  = self.coordinates
        p = self.planet

        if not pl.diagnostic:
            # set up plot
            self.f, self.ax = plt.subplots(2,figsize=(9,9))
            self.f.canvas.set_window_title('CLAuDE')
            self.ax[0].contourf(c.lon_plot, c.lat_plot, c.temperature_world, cmap='seismic')
            self.ax[0].streamplot(c.lon_plot, c.lat_plot, self.u[:,:,0], self.v[:,:,0], color='white',density=1)
            test = self.ax[1].contourf(c.heights_plot, c.lat_z_plot, np.transpose(np.mean(low_level.theta_to_t(self.potential_temperature,p.pressure_levels),axis=1))[:pl.top,:], cmap='seismic',levels=15)
            self.ax[1].contour(c.heights_plot,c.lat_z_plot, np.transpose(np.mean(self.u,axis=1))[:pl.top,:], colors='white',levels=20,linewidths=1,alpha=0.8)
            self.ax[1].quiver(c.heights_plot, c.lat_z_plot, np.transpose(np.mean(self.v,axis=1))[:pl.top,:],np.transpose(np.mean(100*self.w,axis=1))[:pl.top,:],color='black')
            plt.subplots_adjust(left=0.1, right=0.75)
            self.ax[0].set_title('Surface temperature')
            self.ax[0].set_xlim(c.lon.min(),c.lon.max())
            self.ax[1].set_title('Atmosphere temperature')
            self.ax[1].set_xlim(c.lat.min(),c.lat.max())
            self.ax[1].set_ylim((p.pressure_levels.max()/100,p.pressure_levels[:pl.top].min()/100))
            self.ax[1].set_yscale('log')
            self.ax[1].set_ylabel('Pressure (hPa)')
            self.ax[1].set_xlabel('Latitude')
            self.cbar_ax = f.add_axes([0.85, 0.15, 0.05, 0.7])
            self.f.colorbar(test, cax=self.cbar_ax)
            self.cbar_ax.set_title('Temperature (K)')
            self.f.suptitle( 'Time ' + str(round(self.t/p.day,2)) + ' days' )

        else:
            # set up plot
            self.f, self.ax = plt.subplots(2,2,figsize=(9,9))
            self.f.canvas.set_window_title('CLAuDE')
            self.ax[0,0].contourf(c.heights_plot, c.lat_z_plot, np.transpose(np.mean(self.u,axis=1))[:pl.top,:], cmap='seismic')
            self.ax[0,0].set_title('u')
            self.ax[0,1].contourf(c.heights_plot, c.lat_z_plot, np.transpose(np.mean(self.v,axis=1))[:pl.top,:], cmap='seismic')
            self.ax[0,1].set_title('v')
            self.ax[1,0].contourf(c.heights_plot, c.lat_z_plot, np.transpose(np.mean(self.w,axis=1))[:pl.top,:], cmap='seismic')
            self.ax[1,0].set_title('w')
            self.ax[1,1].contourf(c.heights_plot, c.lat_z_plot, np.transpose(np.mean(self.atmosp_addition,axis=1))[:pl.top,:], cmap='seismic')
            self.ax[1,1].set_title('atmosp_addition')
            for axis in ax.ravel():
                axis.set_ylim((p.pressure_levels.max()/100,p.pressure_levels[:pl.top].min()/100))
                axis.set_yscale('log')
            self.f.suptitle( 'Time ' + str(round(self.t/p.day,2)) + ' days' )

        if pl.level_plots:
            self.level_divisions = int(np.floor(p.nlevels/pl.nplots))
            self.level_plots_levels = range(p.nlevels)[::self.level_divisions][::-1]

            self.g, self.bx = plt.subplots(pl.nplots,figsize=(9,8),sharex=True)
            self.g.canvas.set_window_title('CLAuDE pressure levels')
            for k, z in zip(range(pl.nplots), self.level_plots_levels): 
                z += 1
                self.bx[k].contourf(c.lon_plot, c.lat_plot, self.potential_temperature[:,:,z], cmap='seismic')
                self.bx[k].set_title(str(p.pressure_levels[z]/100)+' hPa')
                self.bx[k].set_ylabel('Latitude')
            self.bx[-1].set_xlabel('Longitude')

        plt.ion()
        plt.show()
        plt.pause(2)
        
        if not pl.diagnostic:
            self.ax[0].cla()
            self.ax[1].cla()
            if pl.level_plots:
                for k in range(pl.nplots):
                    self.bx[k].cla()     
        else:
            self.ax[0,0].cla()
            self.ax[0,1].cla()   
            self.ax[1,0].cla()
            self.ax[1,1].cla()

        if pl.above:
            self.g, self.gx = plt.subplots(1,3, figsize=(15,5))
            plt.ion()
            plt.show()

    def __plotting_routine(self):
        # tmp references
        p = self.planet
        c = self.coordinates
        pl = self.plotting

        quiver_padding = int(12/p.resolution)

        if pl.plot:
            if self.verbose: before_plot = time.time()
            # update plot
            if not pl.diagnostic:
                
                # field = temperature_world
                field = np.copy(w)[:,:,self.sample_level]
                # field = np.copy(atmosp_addition)[:,:,self.sample_level]
                test = self.ax[0].contourf(c.lon_plot, c.lat_plot, field, cmap='seismic',levels=15)
                self.ax[0].contour(c.lon_plot, c.lat_plot, self.tracer[:,:,self.sample_level], alpha=0.5, antialiased=True, levels=np.arange(0.01,1.01,0.01))
                if self.velocity: self.ax[0].quiver(c.lon_plot[::quiver_padding,::quiver_padding], c.lat_plot[::quiver_padding,::quiver_padding], self.u[::quiver_padding,::quiver_padding,self.sample_level], self.v[::quiver_padding,::quiver_padding,self.sample_level], color='white')
                self.ax[0].set_xlim((c.lon.min(),c.lon.max()))
                self.ax[0].set_ylim((c.lat.min(),c.lat.max()))
                self.ax[0].set_ylabel('Latitude')
                self.ax[0].axhline(y=0,color='black',alpha=0.3)
                self.ax[0].set_xlabel('Longitude')

                ###

                test = self.ax[1].contourf(c.heights_plot, c.lat_z_plot, np.transpose(np.mean(low_level.theta_to_t(self.potential_temperature,p.pressure_levels),axis=1))[:pl.top,:], cmap='seismic',levels=15)
                # test = ax[1].contourf(heights_plot, lat_z_plot, np.transpose(np.mean(atmosp_addition,axis=1))[:top,:], cmap='seismic',levels=15)
                # test = ax[1].contourf(heights_plot, lat_z_plot, np.transpose(np.mean(potential_temperature,axis=1)), cmap='seismic',levels=15)
                self.ax[1].contour(c.heights_plot, c.lat_z_plot, np.transpose(np.mean(self.tracer,axis=1))[:pl.top,:], alpha=0.5, antialiased=True, levels=np.arange(0.001,1.01,0.01))

                if self.velocity:
                    self.ax[1].contour(c.heights_plot,c.lat_z_plot, np.transpose(np.mean(self.u,axis=1))[:pl.top,:], colors='white',levels=20,linewidths=1,alpha=0.8)
                    self.ax[1].quiver(c.heights_plot, c.lat_z_plot, np.transpose(np.mean(v,axis=1))[:pl.top,:],np.transpose(np.mean(-10*self.w,axis=1))[:pl.top,:],color='black')
                self.ax[1].set_title('$\it{Atmospheric} \quad \it{temperature}$')
                self.ax[1].set_xlim((-90,90))
                self.ax[1].set_ylim((p.pressure_levels.max()/100,p.pressure_levels[:pl.top].min()/100))
                self.ax[1].set_ylabel('Pressure (hPa)')
                self.ax[1].set_xlabel('Latitude')
                self.ax[1].set_yscale('log')
                self.f.colorbar(test, cax=self.cbar_ax)
                self.cbar_ax.set_title('Temperature (K)')
                f.suptitle( 'Time ' + str(round(self.t/p.day,2)) + ' days' )
                    
            else:
                self.ax[0,0].contourf(c.heights_plot, c.lat_z_plot, np.transpose(np.mean(self.u,axis=1))[:pl.top,:], cmap='seismic')
                self.ax[0,0].set_title('u')
                self.ax[0,1].contourf(c.heights_plot, c.lat_z_plot, np.transpose(np.mean(self.v,axis=1))[:pl.top,:], cmap='seismic')
                self.ax[0,1].set_title('v')
                self.ax[1,0].contourf(c.heights_plot, c.lat_z_plot, np.transpose(np.mean(self.w,axis=1))[:pl.top,:], cmap='seismic')
                self.ax[1,0].set_title('w')
                self.ax[1,1].contourf(c.heights_plot, c.lat_z_plot, np.transpose(np.mean(self.atmosp_addition,axis=1))[:pl.top,:], cmap='seismic')
                self.ax[1,1].set_title('atmosp_addition')
                for axis in self.ax.ravel():
                    axis.set_ylim((p.pressure_levels.max()/100,p.pressure_levels[:pl.top].min()/100))
                    axis.set_yscale('log')
                f.suptitle( 'Time ' + str(round(self.t/p.day,2)) + ' days' )

            if pl.level_plots:
                for k, z in zip(range(pl.nplots), self.level_plots_levels): 
                    z += 1
                    self.bx[k].contourf(c.lon_plot, c.lat_plot, self.potential_temperature[:,:,z], cmap='seismic',levels=15)
                    self.bx[k].quiver(c.lon_plot[::quiver_padding,::quiver_padding], c.lat_plot[::quiver_padding,::quiver_padding], self.u[::quiver_padding,::quiver_padding,z], self.v[::quiver_padding,::quiver_padding,z], color='white')
                    self.bx[k].set_title(str(round(p.pressure_levels[z]/100))+' hPa')
                    self.bx[k].set_ylabel('Latitude')
                    self.bx[k].set_xlim((c.lon.min(),c.lon.max()))
                    self.bx[k].set_ylim((c.lat.min(),c.lat.max()))               
                self.bx[-1].set_xlabel('Longitude')

        if pl.above and self.velocity:
            self.gx[0].set_title('Original data')
            self.gx[1].set_title('Polar plane')
            self.gx[2].set_title('Reprojected data')
            self.g.suptitle( 'Time ' + str(round(self.t/p.day,2)) + ' days' )

            if pl.pole == 's':
                self.gx[0].set_title('temperature')
                self.gx[0].contourf(c.lon,c.lat[:self.pole_low_index_S],self.potential_temperature[:self.pole_low_index_S,:,pl.above_level])
                
                self.gx[1].set_title('polar_plane_advect')
                self.polar_temps = low_level.beam_me_up(c.lat[:self.pole_low_index_S],c.lon,self.potential_temperature[:self.pole_low_index_S,:,:],self.grids[1],self.grid_lat_coords_S,self.grid_lon_coords_S)
                output = low_level.beam_me_up_2D(c.lat[:self.pole_low_index_S],c.lon,self.w[:self.pole_low_index_S,:,pl.above_level],self.grids[1],self.grid_lat_coords_S,self.grid_lon_coords_S)

                self.gx[1].contourf(self.grid_x_values_S/1E3,self.grid_y_values_S/1E3,output)
                self.gx[1].contour(self.grid_x_values_S/1E3,self.grid_y_values_S/1E3,self.polar_temps[:,:,pl.above_level],colors='white',levels=20,linewidths=1,alpha=0.8)
                self.gx[1].quiver(self.grid_x_values_S/1E3,self.grid_y_values_S/1E3,self.x_dot_S[:,:,pl.above_level],self.y_dot_S[:,:,pl.above_level])
                
                self.gx[1].add_patch(plt.Circle((0,0),p.planet_radius*np.cos(c.lat[self.pole_low_index_S]*np.pi/180.0)/1E3,color='r',fill=False))
                self.gx[1].add_patch(plt.Circle((0,0),p.planet_radius*np.cos(c.lat[self.pole_high_index_S]*np.pi/180.0)/1E3,color='r',fill=False))

                self.gx[2].set_title('south_addition_smoothed')
                # gx[2].contourf(lon,lat[:pole_low_index_S],south_addition_smoothed[:pole_low_index_S,:,above_level])
                self.gx[2].contourf(c.lon,c.lat[:self.pole_low_index_S],self.u[:self.pole_low_index_S,:,pl.above_level])
                self.gx[2].quiver(c.lon[::5],c.lat[:self.pole_low_index_S],self.u[:self.pole_low_index_S,::5,pl.above_level],self.v[:self.pole_low_index_S,::5,pl.above_level])
            else:
                self.gx[0].set_title('temperature')
                self.gx[0].contourf(c.lon,c.lat[self.pole_low_index_N:],self.potential_temperature[self.pole_low_index_N:,:,pl.above_level])
                
                self.gx[1].set_title('polar_plane_advect')
                self.polar_temps = low_level.beam_me_up(c.lat[self.pole_low_index_N:],c.lon,np.flip(self.potential_temperature[self.pole_low_index_N:,:,:],axis=1),self.grids[0],self.grid_lat_coords_N,self.grid_lon_coords_N)
                output = low_level.beam_me_up_2D(c.lat[self.pole_low_index_N:],c.lon,self.atmosp_addition[self.pole_low_index_N:,:,pl.above_level],self.grids[0],self.grid_lat_coords_N,self.grid_lon_coords_N)
                output = low_level.beam_me_up_2D(c.lat[self.pole_low_index_N:],c.lon,self.w[self.pole_low_index_N:,:,pl.above_level],self.grids[0],self.grid_lat_coords_N,self.grid_lon_coords_N)

                self.gx[1].contourf(self.grid_x_values_N/1E3,self.grid_y_values_N/1E3,output)
                self.gx[1].contour(self.grid_x_values_N/1E3,self.grid_y_values_N/1E3,self.polar_temps[:,:,pl.above_level],colors='white',levels=20,linewidths=1,alpha=0.8)
                self.gx[1].quiver(self.grid_x_values_N/1E3,self.grid_y_values_N/1E3,self.x_dot_N[:,:,pl.above_level],self.y_dot_N[:,:,pl.above_level])
                
                self.gx[1].add_patch(plt.Circle((0,0),p.planet_radius*np.cos(c.lat[self.pole_low_index_N]*np.pi/180.0)/1E3,color='r',fill=False))
                self.gx[1].add_patch(plt.Circle((0,0),p.planet_radius*np.cos(c.lat[self.pole_high_index_N]*np.pi/180.0)/1E3,color='r',fill=False))
        
                self.gx[2].set_title('south_addition_smoothed')
                # gx[2].contourf(lon,lat[pole_low_index_N:],north_addition_smoothed[:,:,above_level])
                self.gx[2].contourf(c.lon,c.lat[self.pole_low_index_N:],self.u[self.pole_low_index_N:,:,pl.above_level])
                self.gx[2].quiver(c.lon[::5],c.lat[self.pole_low_index_N:],self.u[self.pole_low_index_N:,::5,pl.above_level],self.v[self.pole_low_index_N:,::5,pl.above_level])
            
        # clear plots
        if plot or above:   plt.pause(0.001)
        if plot:
            if not c.diagnostic:
                self.ax[0].cla()
                self.ax[1].cla()
                self.cbar_ax.cla()
                        
            else:
                self.ax[0,0].cla()
                self.ax[0,1].cla()
                self.ax[1,0].cla()
                self.ax[1,1].cla()
            if pl.level_plots:
                for k in range(pl.nplots):
                    self.bx[k].cla() 
            if self.verbose:     
                time_taken = float(round(time.time() - before_plot,3))
                print('Plotting: ',str(time_taken),'s') 
        if pl.above:
            self.gx[0].cla()
            self.gx[1].cla()
            self.gx[2].cla()