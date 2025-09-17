import os
import sys
import numpy as np
import healpy as hp
#import healsparse
import scipy as scipy
from scipy.spatial import SphericalVoronoi
from sklearn.neighbors import BallTree
import matplotlib.pyplot as plt
from tqdm import tqdm as tqdm
sys.path.append('./')
import utils as utils
import h5py


class tunnel_finder():
    '''a class to handle higher order statistics analysis on healpix maps
    '''
    
    def __init__( self, input_map, smooth = False, map_has_mask = False, lonlat = True, mask_dominated = False, return_mask_boundary = False, bad_val = hp.UNSEEN ):
        '''
        '''
        #should be able to update this with a hp query for correct generalisation
        
        self.smooth = smooth
        self.map_has_mask = map_has_mask
        self.lonlat = lonlat    
        self.mask_dominated = mask_dominated
        self.return_mask_boundary = return_mask_boundary
        
        self.peak_catalogues_exist = False
        
        self.bad_val = bad_val

        if type(input_map) is str:
            self.load(input_map, self.smooth)
            print("loaded kappa map")   
        elif type(input_map) is np.ndarray:
            self.assign_kappa_map(input_map)
        else:
            raise TypeError('input_map must be either a string or numpy array')
        
        kappa_map = self.get_assigned_kappa_map()
        
        if self.map_has_mask:
            kappa_map = hp.ma(kappa_map)
            self.assign_kappa_map(kappa_map)
            
        self.nside = hp.get_nside(kappa_map)
        
        self.ipix = hp.nside2npix(self.nside)
        self.ipix_arr = np.arange(self.ipix) #array of size npix
        
        self.where_pixel_not_masked = hp.mask_good(kappa_map)
        if self.map_has_mask:
            self.where_pixel_is_masked = kappa_map.mask #get all pixel indices that are masked
        else:
            self.where_pixel_is_masked = []

        self.unmasked_pixels = self.ipix_arr[self.where_pixel_not_masked] #get all pixel indices that are not masked
        self.masked_pixels = self.ipix_arr[self.where_pixel_is_masked]
        
        self.neighbours_of_unmasked_pixels_exists = False
        
        self.assign_kappa_map(kappa_map)
        
        self.peaks_found = False
        
        
        
    def assign_kappa_map(self,kappa_map):
        
        if self.smooth:
            self.kappa_map_smooth = kappa_map
        else:
            self.kappa_map = kappa_map
            
    def get_assigned_kappa_map(self):
        
        if self.smooth:
            return self.kappa_map_smooth
        else:
            return self.kappa_map
        
    def load(self, file_location, smooth = False):
        
        self.smooth = smooth
        
        kappa_map = hp.read_map(file_location)
        
        self.assign_kappa_map(kappa_map)

    
    def smooth_map(self, sl, smoothalm = False, save_location = None ):
        '''sl input is in arcmin
        '''
        ### Need to add option to use hp.smooth alm
        
        self.smooth = True
        self.sl = sl
        self.sl_rad = self.sl / 60. / 180.*np.pi        
        
        if smoothalm:
            self.klm = hp.map2alm(self.kappa_map, pol=False)
            self.klm_smooth = hp.smoothalm( self.klm, sigma = self.sl_rad )
            self.kappa_map_smooth = hp.alm2map( self.klm_smooth, self.nside )
        
        if not smoothalm:
            self.kappa_map_smooth = hp.smoothing( self.kappa_map, sigma = self.sl_rad )
        
        if save_location is not None:
            hp.write_map( save_location, self.kappa_map_smooth )

        
    def get_pdf( self, lower = None,upper = None, nbins = None, density = True ):
        '''
        '''
        
        kappa_map = self.get_assigned_kappa_map()
        
        if lower is None:
            lower = kappa_map.min()
        if upper is None:
            upper = kappa_map.max()
        if nbins is None:
            nbins = 20
            
        self.pdf_lower = lower
        self.pdf_upper = upper
        self.pdf_nbins = nbins
        self.pdf_density = density
        
        self.pdf_binMid, self.pdf_binEdges = utils.get_bins( self.pdf_lower, self.pdf_upper, self.pdf_nbins, return_edges = True)
        
        
        
        self.pdf_counts, discard = np.histogram( kappa_map, density = self.pdf_density, bins = self.pdf_binEdges )

        return self.pdf_counts, self.pdf_binMid

    def find_extrema( self, minima=False, trim_edge_scale = 0 ):
        
        """find extrema in a smoothed masked healpix map
           default is to find peaks, finds minima with minima=True

           Parameters
           ----------
           kappa_masked_smooth: MaskedArray (healpy object)
               smoothed masked healpix map for which extrema are to be identified
           minima: bool
               if False, find peaks. if True, find minima

           Returns
           -------
           extrema_pos: np.ndarray
               extrema positions on sphere, theta and phi, in radians
           extrema_amp: np.ndarray
               extrema amplitudes in kappa

        """

        trim_edge_scale *= np.pi/180.
        #first create an array of all neighbours for all valid healsparse pixels
        #nside = hp.get_nside(kappa_map) #get nside
        
        if not minima:
            print('finding peaks')
        else:
            print('finding minima')
        
        kappa_map = self.get_assigned_kappa_map()
        
         #return true where pixel is not masked
        
        #this part exists so that we don't compute it twice if we don't need to. This saves memory. 
        #i.e. if peaks are identified first, and then minima are identified
        #does this array even need to be assigned to self? 
        #assigning to self uses up much more memory, but speeds up the second extrema computation.
        #should I add an additional flag for if the user will calculate both peaks and minima? 
        #set the default to only one, but provide additional flag for the option if the memory allows it?
        if not self.neighbours_of_unmasked_pixels_exists:
            self.neighbours_of_unmasked_pixels = hp.get_all_neighbours(self.nside, self.unmasked_pixels) #find neighbours for all unmasked pixels
            self.neighbours_of_unmasked_pixels_exists = True
        
        #ipix = np.arange(hp.nside2npix(self.nside))[self.kappa_map.mask==False] #list all pixels and remove masked ones
        #neighbours = hp.get_all_neighbours(self.nside, ipix) #find neighbours for all pixels we care about

        #get kappa values for each pixel in the neighbour array
        neighbour_vals = kappa_map[self.neighbours_of_unmasked_pixels.T]
        #get kappa values for all valid healsparse pixels
        pixel_val = kappa_map[self.unmasked_pixels]

        #compare all valid healsparse pixels with their neighbours to find extrema
        #this line can probably be replaced with something like np.less, may save up to a factor of 8 in memory
        #is it faster to replace np.all with a summation?
        #if minima:
        #    extrema = np.all(np.tile(pixel_val,[8,1]).T < neighbour_vals,axis=-1)
        #else:
        #    extrema = np.all(np.tile(pixel_val,[8,1]).T > neighbour_vals,axis=-1)

        if minima:
            extrema = np.all( np.less(pixel_val[:,None], neighbour_vals), axis=-1)
        else:
            extrema = np.all( np.greater(pixel_val[:,None], neighbour_vals), axis=-1 )
            

        #print the number of extrema identified
        if minima:
            print(f'number of minima identified: {np.where(extrema)[0].shape[0]}')
        else:
            print(f'number of peaks identified: {np.where(extrema)[0].shape[0]}')
        
        print('calculating extrema positions and amplitudes')
        extrema_pos = np.asarray( hp.pix2ang( self.nside, self.unmasked_pixels[extrema], lonlat=self.lonlat ) ).T #find the extrema positions
        extrema_amp = kappa_map[self.unmasked_pixels][extrema]#.data #find the extrema amplitudes
        print('done')
        
        #extrema_amp = np.asarray(extrema_amp)
        
        if trim_edge_scale > 0:
            print('trimming edge extrema')
            #if self.return_mask_boundary:
            #    mask_filter, mask_boundary = self.trim_edge( extrema_pos, trim_edge_scale, self.smooth )
            #else:
            #    mask_filter = self.trim_edge( extrema_pos, trim_edge_scale, smooth )
            mask_filter = self.trim_edge( extrema_pos, trim_edge_scale )
            extrema_pos = extrema_pos[mask_filter]
            extrema_amp = extrema_amp[mask_filter]
        
        if minima:
            self.minima_pos = extrema_pos
            self.minima_amp = extrema_amp
            
            return self.minima_pos, self.minima_amp
            
        else:
            self.peak_pos = extrema_pos
            self.peak_amp = extrema_amp
            
            self.peaks_found = True
            
            return self.peak_pos, self.peak_amp
            

    
    
    def get_extrema_abundance( self, lower = None, upper = None, nbins = None, density = False, minima=False):
        '''
        '''
        
        kappa_map = self.get_assigned_kappa_map()
        
        if minima:
            if lower is not None:
                self.MA_lower = lower
            if lower is None:
                self.MA_lower = kappa_map.min()
                
            if upper is not None:
                self.MA_upper = upper
            if upper is None:
                self.MA_upper = kappa_map.max()
                
            if nbins is not None:
                self.MA_nbins = nbins
            if nbins is None:
                self.MA_nbins = 20
                
            self.MA_density = density

            self.MA_binMid, self.MA_binEdges = utils.get_bins( self.MA_lower, self.MA_upper, self.MA_nbins, return_edges = True)

            

            self.MA_counts, discard = np.histogram( self.minima_amp, density = self.MA_density, bins = self.MA_binEdges )

            return self.MA_counts, self.MA_binMid   
        
        else:
            if lower is not None:
                self.PA_lower = lower
            if lower is None:
                self.PA_lower = kappa_map.min()
                
            if upper is not None:
                self.PA_upper = upper
            if upper is None:
                self.PA_upper = kappa_map.max()
            
            if nbins is None:
                self.PA_nbins = 20
            if nbins is not None:
                self.PA_nbins = nbins
                
            self.PA_density = density

            self.PA_binMid, self.PA_binEdges = utils.get_bins( self.PA_lower, self.PA_upper, self.PA_nbins, return_edges = True)

            kappa_map = self.get_assigned_kappa_map()

            self.PA_counts, discard = np.histogram( self.peak_amp, density = self.PA_density, bins = self.PA_binEdges )

            return self.PA_counts, self.PA_binMid
    
    def get_peak_catalogues(self, thresholds = [None]):
        
        self.N_catalogues = len(thresholds)
        
        self.peak_catalogue_flags = np.zeros([len(thresholds), self.peak_amp.shape[0]], dtype=bool)
        self.thresholds = thresholds
        
        for i, cut_i in enumerate(thresholds):
            if cut_i is not None:
                self.peak_catalogue_flags[i] = self.peak_amp > cut_i
                if not self.peak_catalogues_exist:
                    self.peak_catalogues_exist = True
            else:
                self.peak_catalogue_flags[i] = np.ones(self.peak_amp.shape[0], dtype=bool)
        
        
    def find_tunnels(self,peak_pos=None, overlap=True, trim_mask_frac = None, trim_boundary=False, N_void_rad_trim = 1, angle_trim = 0, catalogue_index = None, return_boundary_points = False):
        """Finds tunnels for a specific catalogue as defined by get_peak_catalogues
        """
        if not self.peaks_found and peak_pos is None:
            self.find_extrema()
        if peak_pos is not None:
            self.peak_pos = peak_pos
            self.peak_amp = np.zeros([peak_pos.shape[0]])
        
        print(f'peak position shape: {self.peak_pos.shape}')
        
        radius = 1 #define a unit sphere
        center = np.array([0, 0, 0]) #with center at the origin
        
        if catalogue_index is not None:
            catalogue_filter = self.peak_catalogue_flags[catalogue_index]
        else:
            catalogue_filter = np.ones( [len(self.peak_amp)], dtype=bool )
            
        peak_pos_filtered = self.peak_pos[catalogue_filter]
        
        print('calculating point vectors')
        points = hp.ang2vec( peak_pos_filtered[:,0], peak_pos_filtered[:,1], lonlat=self.lonlat ) #convert peak angular positions to 3d xyz vectors
        
        print('constructing Voronoi tesselation')
        sv = SphericalVoronoi( points, radius, center ) #construct vornoi tesselation on sphere surface from peaks
        
        print('calculating circumcenter angular positions')
        circumcenters = np.array(hp.vec2ang(sv.vertices,lonlat=self.lonlat)).T #delaunay circumcenters are equal to the vornoi vertices
        
        #print('lonlat to latlon')
        peak_pos_latlon = np.column_stack( [ peak_pos_filtered[:,1], peak_pos_filtered[:,0] ] ) #healpy uses lonlat in degrees. balltree uses latlon in radians
        circumcenters_latlon = np.column_stack([circumcenters[:,1],circumcenters[:,0]]) #swap columns again as with peak_pos.
        
        print('{0} voids found'.format(circumcenters.shape[0]))
        
        print('constructing peak position balltree')
        #can ballTree functions be replaced with healpy? I initialise lots of trees throughout the code which have noticeable overheads.
        self.btp = BallTree(np.deg2rad(peak_pos_latlon), metric='haversine') #build BallTree for quick point distance calculations, haversine metric for surface of sphere
        distances, indices = self.btp.query(np.deg2rad(circumcenters_latlon),k=1) #find distances from each circumcenter to nearest peak, this is the same as the radius of the circumcircle
        #assign above variables to self? This can speed up the angle calculations later on. 
        distances = distances[:,0]
        
        if return_boundary_points:
            _, peak_boundary_indices = self.btp.query(np.deg2rad(circumcenters_latlon),k=3)
        
        #print('finding void sizes')
        #distances = hp.rotator.angdist(peak_pos_filtered, sv.regions ,lonlat=True)
        
        
        print('sorting voids')
        sort = np.argsort(distances)[::-1] #reverse order so largest void is first

        circumcenters_sorted = circumcenters[sort]
        circumcenters_latlon_sorted = circumcenters_latlon[sort] #again, balltree 
        distances_sorted = distances[sort]
        if return_boundary_points:
            peak_boundary_indices_sorted = peak_boundary_indices[sort]
        
        
        if trim_boundary:
            print('trimming boundary voids')
            trim_edge_voids_filter = self.trim_edge(circumcenters_sorted, N_void_rad_trim * distances_sorted)
            N_removed = len(distances_sorted) - trim_edge_voids_filter.sum()
            print(f"{N_removed} voids near mask boundary")
        if not trim_boundary:
            trim_edge_voids_filter = np.ones(len(distances_sorted),dtype=bool)

        circumcenters_sorted = circumcenters_sorted[trim_edge_voids_filter]
        circumcenters_latlon_sorted = circumcenters_latlon_sorted[trim_edge_voids_filter]
        distances_sorted = distances_sorted[trim_edge_voids_filter]
        if return_boundary_points:
            peak_boundary_indices_sorted = peak_boundary_indices_sorted[trim_edge_voids_filter]
            
        if trim_mask_frac is not None:
            print('trimming mask fraction')
            trim_mask_filter = self.trim_mask_fraction(trim_mask_frac,circumcenters_sorted, distances_sorted)
            
            circumcenters_sorted = circumcenters_sorted[trim_mask_filter]
            circumcenters_latlon_sorted = circumcenters_latlon_sorted[trim_mask_filter]
            distances_sorted = distances_sorted[trim_mask_filter]
            if return_boundary_points:
                peak_boundary_indices_sorted = peak_boundary_indices_sorted[trim_mask_filter]
            
            print(f'{len(trim_mask_filter) - trim_mask_filter.sum()} mask fraction voids removed')
            

        print('trimming small angle voids')
        #need to see if I can speed up this part of the code too
        if angle_trim > 0:
            small_angles_filter = self.get_small_angles_filter(peak_pos_latlon,circumcenters_latlon_sorted,angle_trim)
        if angle_trim == 0:
            small_angles_filter = np.ones(len(distances_sorted),dtype=bool)


        circumcenters_sorted = circumcenters_sorted[small_angles_filter]
        circumcenters_latlon_sorted = circumcenters_latlon_sorted[small_angles_filter]
        distances_sorted = distances_sorted[small_angles_filter]
        if return_boundary_points:
            peak_boundary_indices_sorted = peak_boundary_indices_sorted[small_angles_filter]
            

        print('trimming overlapping voids')
        if not overlap:        
            overlap_filter = self.remove_overlap(circumcenters_latlon_sorted, distances_sorted)
        if overlap:
            overlap_filter = np.ones(len(distances_sorted),dtype=bool)

        circumcenters_sorted = circumcenters_sorted[overlap_filter]
        circumcenters_latlon_sorted = circumcenters_latlon_sorted[overlap_filter]
        distances_sorted = distances_sorted[overlap_filter]     
        if return_boundary_points:
            peak_boundary_indices_sorted = peak_boundary_indices_sorted[overlap_filter]

        #final_filter = trim_edge_voids_filter * small_angles_filter * overlap_filter

        #N_total_removed = len(distances_sorted) - final_filter.sum()
        #print(f'{N_total_removed} voids removed in total') #e.g. some voids may have a small angle and be on the edge, but cannot be reomoved twice
        #final_filter = final_filter.astype(dtype=bool)
        #print(f'{len(distances_sorted)} voids found in total after filtering')

        #return circumcenters_sorted[final_filter], distances_sorted[final_filter]


        print(f'{distances_sorted.shape[0]} voids in final catalogue')
        if return_boundary_points:
            self.peak_boundary_indices = peak_boundary_indices_sorted
        self.tunnel_positions = circumcenters_sorted 
        self.tunnel_radii = distances_sorted * 180./np.pi #convert from radians to degrees
        
        if return_boundary_points:
            return self.tunnel_positions, self.tunnel_radii, self.peak_boundary_indices
        else:
            return self.tunnel_positions, self.tunnel_radii
    
    
    def get_tunnel_catalogues(self, overlap=True, trim_mask_frac = None, trim_boundary=False, N_void_rad_trim = 1, angle_trim = 0, return_boundary_points = False):
        
        if not self.peak_catalogues_exist:
            raise ValueError('peak catalogues have not been generated, see get_peak_catalogues()')
        
        self.tunnel_positions_catalogues = [] #should I just use lists here?
        self.tunnel_radii_catalogues = [] 
        if return_boundary_points:
            self.tunnel_boundary_points_catalogues = []
        
        for i in range(len(self.thresholds)):
            hold = self.find_tunnels(overlap=overlap, trim_mask_frac = trim_mask_frac, trim_boundary=trim_boundary, N_void_rad_trim=N_void_rad_trim, angle_trim=angle_trim, catalogue_index = i, return_boundary_points = return_boundary_points)
            
            self.tunnel_positions_catalogues.append([hold[0]]) 
            self.tunnel_radii_catalogues.append([hold[1]])
        
            if return_boundary_points:
                self.tunnel_boundary_points_catalogues.append([hold[2]]) 
            
        if return_boundary_points: 
            return self.tunnel_positions_catalogues, self.tunnel_radii_catalogues, self.tunnel_boundary_points_catalogues
        else:
            return self.tunnel_positions_catalogues, self.tunnel_radii_catalogues

    
    def remove_overlap(self,void_pos,void_rad):

        btc = BallTree(np.deg2rad(void_pos),metric='haversine') #build tree

        N_voids_start = void_rad.shape[0]
        accepted_voids = np.zeros(N_voids_start,dtype=bool)        
        queue = np.arange(0,N_voids_start)

        progress = tqdm()
        while len(queue) > 0:

            candidate = queue[0]
            accepted_voids[candidate] = True
            queue = queue[1:] #remove accepted void
            #remove candidate sub voids from queue:
            delete_list = btc.query_radius((np.deg2rad(void_pos)[candidate],),(void_rad[candidate],))[0] 
            queue = np.setdiff1d( queue, delete_list, assume_unique = True )

            progress.update()

        print(f'{N_voids_start - accepted_voids.sum()} overlapping voids removed')

        return accepted_voids

    def remove_overlap_fast(self,void_pos,void_rad):
        print('using faster function')
        btc = BallTree(np.deg2rad(void_pos),metric='haversine') #build tree
        sub_void_indices = btc.query_radius(np.deg2rad(void_pos),void_rad) #find all voids within a void including the void. Test if the void is included with sub voids
        N_voids_start = void_rad.shape[0]
        accepted_voids = np.ones(N_voids_start,dtype=bool)  # initially all voids are accepted
        for i in range(N_voids_start):
            if accepted_voids[i] == 0: # if a void hasn't been accepted, do nothing and go to next one
                continue
            accepted_voids[sub_void_indices[i]] = 0 # if a void is accepted, then all the ones that overlap with it are not accepted
            accepted_voids[i] = 1 # we mark the i-th void as accepted (noting the above line includes the i-th void, which was set to 0)
                                
        print(f'{N_voids_start - accepted_voids.sum()} overlapping voids removed')

        return accepted_voids
    

    def get_angles(self,p,coords):
        '''p is a list of triangle vertices
        '''

        if coords == 'lonlat':
            c = hp.ang2vec(p[:,0],p[:,1],lonlat=True)
        if coords == 'latlon':
            c = hp.ang2vec(p[:,1],p[:,0],lonlat=True )

        v01 = c[1] - c[0]
        v12 = c[2] - c[1]
        v02 = c[2] - c[0]
        
        v01_mod = scipy.linalg.norm(v01)
        v12_mod = scipy.linalg.norm(v12)
        v02_mod = scipy.linalg.norm(v02)

        a0 = np.arccos( np.dot(v02,v01) / ( v02_mod * v01_mod ) )
        a1 = np.pi - np.arccos( np.dot(v01,v12) / ( v01_mod * v12_mod ) ) # subtract from pi because vectors do not have common origin or end point
        a2 = np.arccos( np.dot(v12,v02) / ( v12_mod * v02_mod ) )


        angles = np.rad2deg(np.array([a0,a1,a2]))

        return angles
    

    def get_small_angles_filter(self,peak_pos,circumcenters,angle_trim):
        #btp = BallTree(np.deg2rad(peak_pos), metric='haversine') #build BallTree for quick point distance calculations, haversine metric for surface of sphere
        distances, indices = self.btp.query(np.deg2rad(circumcenters),k=3) #find distances from each circumcenter to nearest 3 peaks, this is the same as the radius of the circumcircle
        #The above tree has already been computed when finding the tunnels. I should optimise this. 
        N_voids = indices.shape[0]

        #can I speed up this for loop with numpy arrays?
        angles = np.zeros((indices.shape))
        #for j in tqdm(range(N_voids)):
        for j in range(N_voids):
            angles[j] = self.get_angles(peak_pos[indices[j]],coords='latlon')

        small_angles_filter = angles.min(axis=1) > angle_trim #false when angle is too small compared to angle_trim 
        N_removed = N_voids - np.sum(small_angles_filter) 
        print( f'{N_removed} voids with small angles' )

        return small_angles_filter
        
        
    def trim_edge( self, pos, trim_scale ):
        'trim scale is in radians here'

        kappa_map = self.get_assigned_kappa_map()

        #is_masked = np.where( kappa_map == self.bad_val ) #find all masked pixels
        is_masked = kappa_map.mask
        
        pixel_masked = self.ipix_arr[is_masked] #filter out non masked pixels from pixel list

        if len(pixel_masked)==0:
            return np.ones([pos.shape[0]], dtype=bool)
        
        pixel_ang_trim = np.array( hp.pix2ang(self.nside, pixel_masked, lonlat=self.lonlat ) ).T #angular coordinates of masked pixels

        factor = np.pi / 180.
        btt = BallTree(pixel_ang_trim[:,::-1] * factor, metric='haversine')

        distances, indices = btt.query(pos[:,::-1] * factor, k=1) #[:,::-1] because btt.query takes lat lon and hp uses lon lat.

        mask_filter = distances[:,0] > trim_scale

        #t1 = time.time()
        #total = t1-t0
        #print(f'time to compute = {total}')

        if self.return_mask_boundary: #note here that self.return_mask_boundary returns the whole mask now
            return mask_filter, pixel_ang_trim
        else:
            return mask_filter
        
    def trim_mask_fraction(self, mask_fraction, pos, rad):
        
        kappa_map = self.get_assigned_kappa_map()

        #is_masked = np.where( kappa_map == self.bad_val ) #find all masked pixels
        is_masked = kappa_map.mask
        
        pixel_masked = self.ipix_arr[is_masked] #filter out non masked pixels from pixel list

        if len(pixel_masked)==0:
            return np.ones([pos.shape[0]], dtype=bool)
        
        pixel_ang_trim = np.array( hp.pix2ang(self.nside, pixel_masked, lonlat=self.lonlat ) ).T #angular coordinates of masked pixels

        factor = np.pi / 180.
        btt = BallTree(pixel_ang_trim[:,::-1] * factor, metric='haversine')

        indices = btt.query_radius(pos[:,::-1] * factor, rad) #[:,::-1] because btt.query takes lat lon and hp uses lon lat.

        
        N_void = rad.shape[0]
        mask_filter = np.zeros(N_void) 
        
        for i in range(N_void):
           
            mask_filter[i] = len(indices[i]) * hp.nside2pixarea(self.nside)
        
        mask_filter /= (np.pi * rad**2)

        mask_filter_bool = mask_filter < mask_fraction
        
        return mask_filter_bool

        
        
    def get_profiles(self, nbins, r_min, r_max, void_pos=None, void_rad=None, return_all = False, inclusive=False, weights=True ):

        #also need functionality for if we are looking at void catalogues
        #define a get_statistic function that gets either the catalogue or none catalogue version of the statistics
        
        if void_pos is None:
            void_pos = self.tunnel_positions
        if void_rad is None:
            void_rad = self.tunnel_radii
        
        
        N_voids = void_rad.shape[0] #should this be an attribute?
        r_mid, r_edge = utils.get_bins(r_min, r_max, nbins, return_edges = True)
        kappa_map = self.get_assigned_kappa_map()

        void_kappa_profiles = np.zeros([N_voids,nbins])
        
        void_pos_vec = hp.ang2vec(theta = void_pos[:,0], phi = void_pos[:,1], lonlat=self.lonlat)
        
        for i in tqdm(range(N_voids)):
            for j in range(nbins):
                
                outer_pixels = hp.query_disc( self.nside, void_pos_vec[i], np.deg2rad(void_rad[i])*r_edge[j+1], inclusive=inclusive )
                
                if j == 0:
                    void_kappa_profiles[i,j] = kappa_map[outer_pixels].mean()
                    inner_pixels = outer_pixels
                    continue

                annuli = np.setxor1d(inner_pixels, outer_pixels, assume_unique=True)

                void_kappa_profiles[i,j] = kappa_map[annuli].mean()
                
                inner_pixels = outer_pixels
                
        print('taking weighted average')
        
        void_kappa_profiles_filled = utils.fill_nan(void_kappa_profiles)
        self.void_kappa_profiles = void_kappa_profiles
        if weights:
            weights = void_rad**2
        else:
            weights = None
        kappa_mean_weighted = np.average(void_kappa_profiles_filled, axis=0, weights=weights)
        
        if return_all:
            return r_mid, void_kappa_profiles_filled, kappa_mean_weighted
        else:
            return r_mid, kappa_mean_weighted
        
        
    def get_profiles_cross_map(self, cross_map, nbins, r_min, r_max, void_pos=None, void_rad=None, weights=True, return_all = False, inclusive=False, cross_kappa=False, local_background_subtraction = False):

        #also need functionality for if we are looking at void catalogues
        #define a get_statistic function that gets either the catalogue or none catalogue version of the statistics
        
        if void_pos is None:
            void_pos = self.tunnel_positions
        if void_rad is None:
            void_rad = self.tunnel_radii
        
        
        N_voids = void_rad.shape[0] #should this be an attribute?
        r_mid, r_edge = utils.get_bins(r_min, r_max, nbins, return_edges = True)
        #kappa_map = cross_map
        cross_map_nside = hp.get_nside(cross_map)
        
        
        void_kappa_profiles = np.zeros([N_voids,nbins])
        
        void_pos_vec = hp.ang2vec(theta = void_pos[:,0], phi = void_pos[:,1], lonlat=self.lonlat)
        
        if not cross_kappa:
            pixel_area = hp.nside2pixarea(cross_map_nside, degrees=True)
        else:
            pixel_area = 1
        print(f'cross kappa flag = {cross_kappa}')
        print(f'pixel area = {pixel_area}')
        print(f'local background subtraction = {local_background_subtraction}')
        cross_map_mean = np.nanmean(cross_map) #/ pixel_area
        print(f'cross map mean = {cross_map_mean}')
        print(f'cross map median = {np.nanmedian(cross_map)}')
        if not cross_kappa:
            if not local_background_subtraction:
                print('creating overdensity map')
                #cross_map /= pixel_area
                cross_map -= cross_map_mean
                cross_map /= cross_map_mean

        kappa_map = cross_map
        print(f'mean after scaling = {np.nanmean(kappa_map)}')
        
        local_mean = np.zeros(void_kappa_profiles.shape)
        
        for i in tqdm(range(N_voids)):
            for j in range(nbins):
                
                outer_pixels = hp.query_disc( cross_map_nside, void_pos_vec[i], np.deg2rad(void_rad[i])*r_edge[j+1], inclusive=inclusive )
                
                if j == 0:
                    void_kappa_profiles[i,j] = np.nanmean(kappa_map[outer_pixels])# / (np.count_nonzero(~np.isnan(kappa_map[outer_pixels])) * pixel_area) #kappa_map[outer_pixels].sum() / (len(outer_pixels)*pixel_area)
                    inner_pixels = outer_pixels
                    continue

                annuli = np.setxor1d(inner_pixels, outer_pixels, assume_unique=True)                
                
                void_kappa_profiles[i,j] = np.nanmean(kappa_map[annuli]) #/ (np.count_nonzero(~np.isnan(kappa_map[annuli])) * pixel_area) #kappa_map[annuli].sum() / (len(annuli)*pixel_area)
                
                inner_pixels = outer_pixels
                
            if local_background_subtraction:
                if i==0 and j==0: 
                    print('performing local background subtraction')
                outer_pixels = hp.query_disc( cross_map_nside, void_pos_vec[i], np.deg2rad(void_rad[i])*r_edge[-1], inclusive=inclusive )
                inner_pixels = hp.query_disc( cross_map_nside, void_pos_vec[i], np.deg2rad(void_rad[i])*r_edge[-2], inclusive=inclusive )
                annuli = np.setxor1d(inner_pixels, outer_pixels, assume_unique=True)
                local_mean[i,:] = np.nanmean(kappa_map[annuli])
                
                #void_kappa_profiles[i] -= local_mean
                #void_kappa_profiles[i] /= local_mean
                
        if local_background_subtraction:
            print(f'subtracting local means: {local_mean}')
            void_kappa_profiles -= local_mean
        
        print('taking weighted average')
        print(f'cross map mean = {cross_map_mean}')
        #if not cross_kappa:
        #    void_kappa_profiles -= cross_map_mean
        #    void_kappa_profiles /= cross_map_mean
        
        
        fully_masked_filter = ~np.all(np.isnan(void_kappa_profiles), axis=1)

        void_kappa_profiles_filled = utils.fill_nan(void_kappa_profiles[fully_masked_filter])

        self.void_kappa_profiles = void_kappa_profiles
        
        if weights:
            weights_in = void_rad[fully_masked_filter]**2
        else:
            weights_in = None
            
        #if not bool(int(np.ceil(cross_map_mean))): #what is this line here for?
        #    kappa_mean_weighted = np.zeros([len(r_mid)])
        #else:
        kappa_mean_weighted = np.average(void_kappa_profiles_filled, axis=0, weights=weights_in)
        
        
        
        if return_all:
            return r_mid, void_kappa_profiles_filled, weights_in, fully_masked_filter
        else:
            return r_mid, kappa_mean_weighted
        
        
    def get_profiles_cross_kappa_map(self, cross_map, nbins, r_min, r_max, void_pos=None, void_rad=None, weights=True, return_all = False, inclusive=False):
        
        return self.get_profiles_cross_map(cross_map, nbins, r_min, r_max, void_pos=void_pos, void_rad=void_rad, weights=weights, return_all = return_all, inclusive=inclusive, cross_kappa=True)

    
    
    
    
    
    def get_profiles_cross_map_CTHFilter(self, cross_map, nbins, r_min, r_max, void_pos=None, void_rad=None, weights=True, return_all = False, inclusive=False):

        if void_pos is None:
            void_pos = self.tunnel_positions
        if void_rad is None:
            void_rad = self.tunnel_radii
        
        
        N_voids = void_rad.shape[0] #should this be an attribute?
        r_mid, r_edge = utils.get_bins(r_min, r_max, nbins, return_edges = True)
        cross_map_nside = hp.get_nside(cross_map)
        
        void_kappa_profiles = np.zeros([N_voids,nbins])
        void_pos_vec = hp.ang2vec(theta = void_pos[:,0], phi = void_pos[:,1], lonlat=self.lonlat)

        kappa_map = cross_map
        
        pixel_area = hp.nside2pixarea(cross_map_nside)
        pixel_width = np.sqrt( pixel_area )
        
        for i in tqdm(range(N_voids)):
            for j in range(nbins):
                

                
                R_in = np.deg2rad(void_rad[i]) * r_mid[j]
                R_out = np.sqrt(2) * np.deg2rad(void_rad[i]) * r_mid[j]
                
                
                inner_pixels = hp.query_disc( cross_map_nside, void_pos_vec[i], R_in, inclusive=inclusive )
                
                outer_pixels = hp.query_disc( cross_map_nside, void_pos_vec[i], R_out, inclusive=inclusive )
                
                outer_annuli = np.setxor1d(inner_pixels, outer_pixels, assume_unique=True)   
                
                #test 1 - nearly works. factor of 1e6 or 1e7 too large
                #inner_term = ( utils.nansumwrapper(kappa_map[inner_pixels])  ) / ( R_in )
                #outer_term = ( utils.nansumwrapper(kappa_map[outer_annuli])  ) / ( R_out - R_in )
                
                #test 2 - doesn't work? - try for all cuts - seems like the only reliable one now
                inner_term = np.nanmean(kappa_map[inner_pixels]) 
                outer_term = np.nanmean(kappa_map[outer_annuli])
                
                #test 3 - works?
                #inner_term = np.nanmean(kappa_map[inner_pixels]) * ( R_in )
                #outer_term = np.nanmean(kappa_map[outer_annuli]) * ( R_out + R_in )
                
                #test 4 - gives same answer as 3
                #inner_term = ( utils.nansumwrapper(kappa_map[inner_pixels]) * pixel_area ) / ( R_in )
                #outer_term = ( utils.nansumwrapper(kappa_map[outer_annuli]) * pixel_area ) / ( R_out - R_in )
                
                
                void_kappa_profiles[i,j] = inner_term - outer_term
                    

        fully_masked_filter = ~np.all(np.isnan(void_kappa_profiles), axis=1)

        void_kappa_profiles_filled = utils.fill_nan(void_kappa_profiles[fully_masked_filter])

        self.void_kappa_profiles = void_kappa_profiles
        
        if weights:
            weights_in = void_rad[fully_masked_filter]**2
        else:
            weights_in = None
            

        kappa_mean_weighted = np.average(void_kappa_profiles_filled, axis=0, weights=weights_in)
        
        
        if return_all:
            return r_mid, void_kappa_profiles_filled, weights_in, fully_masked_filter
        else:
            return r_mid, kappa_mean_weighted
    
    
    
    
    
    

    #The load functions should be a separate class that inherits the HOS properties?
    def load_tunnels(self,load_file_path,realisation,tomo):
        
        with h5py.File(load_file_path, 'r') as hf:
            hold = hf[f'tunnel_catalogues/realisation{realisation}/tomo{tomo}'][:]
            self.tunnel_positions = hold[:,:2]
            self.tunnel_radii = hold[:,2]
            
        return self.tunnel_positions, self.tunnel_radii
    
    def load_tunnels_grid(self, load_file_path, grid_i, realisation, tomo):
        
        with h5py.File(load_file_path, 'r') as hf:
            hold = hf[f'tunnel_catalogues/{grid_i}/realisation{realisation}/tomo{tomo}'][:]
            self.tunnel_positions = hold[:,:2]
            self.tunnel_radii = hold[:,2]
            
        return self.tunnel_positions, self.tunnel_radii
    
    def load_extrema(self,load_file_path,realisation,tomo,minima=False):
        
        with h5py.File(load_file_path, 'r') as hf:  
            
            if minima:
                hold = hf[f'minima_catalogues/realisation{realisation}/tomo{tomo}'][:]
                self.minima_pos = hold[:2,:]
                self.minima_amp = hold[2,:]
                
                return self.minima_pos, self.minima_amp
                
            else:
                hold = hf[f'peak_catalogues/realisation{realisation}/tomo{tomo}'][:]
                self.peak_pos = hold[:2,:]
                self.peak_amp = hold[2,:]

                self.peaks_found = True
            
                return self.peak_pos, self.peak_amp
        
        
    def preview(self,size,npix,center, features = [], vmin=-0.02, vmax=0.06):
    #improve input variable names
    #add generality to plot any of the features that have been identified
    #should features be a dictionary?
    
        kappa_map = self.get_assigned_kappa_map()
        map_gnom = hp.gnomview(kappa_map,rot=center,xsize=npix,reso=(size * 60) / npix,
                               no_plot=True,return_projected_map=True)

        R = hp.Rotator(rot=center)
        
        #set marker size
        ms = 2
        
        if not 'peak catalogues' in features or not 'void catalogues' in features:
            
            if 'peaks' in features:
                peak_pos = self.peak_pos

                peak_pos_rot = hp.rotator.rotateDirection(R.mat,peak_pos[:,0],peak_pos[:,1],lonlat=True)

                #trim peaks to only the region we want to plot
                trimP_col = np.abs(peak_pos_rot.T) < size/2
                trimP = trimP_col[:,0] * trimP_col[:,1]

                peak_pos_plot = peak_pos_rot.T[trimP]

                plt.plot(-1*peak_pos_plot[:,0],peak_pos_plot[:,1],'.',color='r',markersize=ms)

            if 'minima' in features:
                min_pos = self.minima_pos

                min_pos_rot = hp.rotator.rotateDirection(R.mat,min_pos[:,0],min_pos[:,1],lonlat=True)

                #trim peaks to only the region we want to plot
                trimM_col = np.abs(min_pos_rot.T) < size/2
                trimM = trimM_col[:,0] * trimM_col[:,1]

                min_pos_plot = min_pos_rot.T[trimM]

                plt.plot(-1*min_pos_plot[:,0],min_pos_plot[:,1],'.',color='c',markersize=ms)

            if 'voids' in features:
                circumcenters = self.tunnel_positions
                distances = self.tunnel_radii

                cc_pos_rot = hp.rotator.rotateDirection(R.mat,circumcenters[:,0],circumcenters[:,1],lonlat=True)

                trimC_col = np.abs(cc_pos_rot.T) < size/2
                trimC = trimC_col[:,0] * trimC_col[:,1]

                cc_pos_plot = cc_pos_rot.T[trimC]

                #print(trimC)

                plt.plot(-1*cc_pos_plot[:,0],cc_pos_plot[:,1],'.',color='g',markersize=ms)

                ax = plt.gca()
                for i in range(distances[trimC].shape[0]):
                    circle = plt.Circle((-1*cc_pos_plot[i,0],cc_pos_plot[i,1]), distances[trimC][i], color='k', fill=False)
                    ax.add_patch(circle)
                    
            plt.imshow(map_gnom,origin='lower',vmin = vmin,vmax = vmax,
                       extent = [-size/2,+size/2,
                                 -size/2,+size/2] )
            plt.show()
            
        if 'peaks catalogues' in features:
                
                for i in range(self.N_catalogues):
                    peak_pos = self.peak_pos[self.peak_catalogue_flags[i]]

                    peak_pos_rot = hp.rotator.rotateDirection(R.mat,peak_pos[:,0],peak_pos[:,1],lonlat=True)

                    #trim peaks to only the region we want to plot
                    trimP_col = np.abs(peak_pos_rot.T) < size/2
                    trimP = trimP_col[:,0] * trimP_col[:,1]

                    peak_pos_plot = peak_pos_rot.T[trimP]

                    plt.plot(-1*peak_pos_plot[:,0],peak_pos_plot[:,1],'.',color='r',markersize=ms)
                    
                    plt.imshow(map_gnom,origin='lower',vmin = vmin,vmax = vmax,
                           extent = [-size/2,+size/2,
                                     -size/2,+size/2] )

                    plt.show()
        
        if 'void catalogues' in features:
            
            for i in range(len(thresholds)):
                circumcenters = self.tunnel_positions_catalogues[i]
                distances = self.tunnel_radii_catalogues[i]
                
                cc_pos_rot = hp.rotator.rotateDirection(R.mat,circumcenters[:,0],circumcenters[:,1],lonlat=True)

                trimC_col = np.abs(cc_pos_rot.T) < size/2
                trimC = trimC_col[:,0] * trimC_col[:,1]

                cc_pos_plot = cc_pos_rot.T[trimC]

                #print(trimC)

                plt.plot(-1*cc_pos_plot[:,0],cc_pos_plot[:,1],'.',color='g',markersize=ms)

                ax = plt.gca()
                for i in range(distances[trimC].shape[0]):
                    circle = plt.Circle((-1*cc_pos_plot[i,0],cc_pos_plot[i,1]), distances[trimC][i], color='k', fill=False)
                    ax.add_patch(circle)

                plt.imshow(map_gnom,origin='lower',vmin = vmin,vmax = vmax,
                       extent = [-size/2,+size/2,
                                 -size/2,+size/2] )

                plt.show()
                
        if 'peak and void catalogues' in features:
            ###...
            for i in range(len(thresholds)):
                circumcenters = self.tunnel_positions_catalogues[i]
                distances = self.tunnel_radii_catalogues[i]
                
                cc_pos_rot = hp.rotator.rotateDirection(R.mat,circumcenters[:,0],circumcenters[:,1],lonlat=True)

                trimC_col = np.abs(cc_pos_rot.T) < size/2
                trimC = trimC_col[:,0] * trimC_col[:,1]

                cc_pos_plot = cc_pos_rot.T[trimC]

                #print(trimC)

                plt.plot(-1*cc_pos_plot[:,0],cc_pos_plot[:,1],'.',color='g',markersize=ms)

                ax = plt.gca()
                for i in range(distances[trimC].shape[0]):
                    circle = plt.Circle((-1*cc_pos_plot[i,0],cc_pos_plot[i,1]), distances[trimC][i], color='k', fill=False)
                    ax.add_patch(circle)

                plt.imshow(map_gnom,origin='lower',vmin = vmin,vmax = vmax,
                       extent = [-size/2,+size/2,
                                 -size/2,+size/2] )

                plt.show()
            
            

