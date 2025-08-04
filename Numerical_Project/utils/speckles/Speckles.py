import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import cv2
import typing
from functools import cache
import pyfftw
from datetime import datetime
pyfftw.interfaces.cache.enable()

matplotlib.use('TKAgg')


class Speckles:
    
    MICRON = 1e-6
    THIN_DIFFUSER =1
    THICK_DIFFUSER =2
    wl = 808e-9 # Wavelength nanometer
    sig = 700e-6 # of the Gaussian 700 micron
    L = 20e-3 # 20 mm 
    # L = 10e-3 
    theta = 0.02 # Desired Scattering angle
    k = 2*np.pi/wl
    active_slice = np.index_exp[450:650, 450:650]
    roi =  np.index_exp[509:512, 509:512]
    
    def __init__(self, Nx, Ny, Dx, Dy) -> None:
        self.diffuser1_phase = None
        self.diffuser2_phase = None
        self.slm_phase = None
        self.E_in = None
        self.is_thick: bool = None
        self.thick_diffuser_width = None
        self.Nx = Nx
        self.Ny = Ny
        self.Dx = Dx
        self.Dy = Dy
        self.dx = self.Dx / self.Nx
        self.dy = self.Dy / self.Ny # pixel size 3 micron
    
        
        # init grid
        X = (np.arange(1, Nx + 1) - (Nx / 2 + 0.5)) * self.dx
        Y = (np.arange(1, Ny + 1) - (Ny / 2 + 0.5)) * self.dy
        self.XX, self.YY = np.meshgrid(X, Y)

        # defing k-space
        fs = 1 / (self.XX.max() - self.XX.min())  # sapatial freq = 1/x
        freq_x = fs * np.arange(-Nx // 2, Nx // 2)
        fs = 1 / (self.YY.max() - self.YY.min())
        freq_y = fs * np.arange(-Ny // 2, Ny // 2)
        # np.fft.fftfreq does this automatically for you 
        freq_XXs, freq_YYs = np.meshgrid(freq_x, freq_y)
        light_k = 2 * np.pi / Speckles.wl
        k_xx = freq_XXs * 2 * np.pi
        k_yy = freq_YYs * 2 * np.pi
        k_z_sqr = light_k ** 2 - (k_xx ** 2 + k_yy ** 2)
        # Remove all the negative component, as they represent evanescent waves, see Fourier Optics page 58
        np.maximum(k_z_sqr, 0, out=k_z_sqr)
        self.k_z = np.sqrt(k_z_sqr)


    def _get_prop_mat(self,l,rev=False):
        if rev == True:
            return np.exp(-1j * self.k_z * l)
        return np.exp(+1j *  self.k_z * l)

    def _my_fft2(self,E):
        return np.fft.fftshift(pyfftw.interfaces.numpy_fft.fft2(
            np.fft.fftshift(E), overwrite_input=False, auto_align_input=True)
    )

    def _my_ifft2(self,E_K):
        return np.fft.fftshift(pyfftw.interfaces.numpy_fft.ifft2(
            np.fft.fftshift(E_K), overwrite_input=False, auto_align_input=True)
    )

    def phase_mask_macro_pixels_method(self, macro_pixel_size)->np.ndarray:
        init_mask = np.random.uniform(0, 2*np.pi, size=( self.Nx//macro_pixel_size,  self.Ny//macro_pixel_size))
        # resize: start with 256X256 random, and then resize to 1024X1024 for macro pixels 
        phase_mask = cv2.resize(init_mask, self.XX.shape[::-1], interpolation=cv2.INTER_AREA)
        return phase_mask


    def free_space_propagation(self,E_init,l):
        E_K = self._my_fft2(E_init) # np.fft.fftshift(np.fft.fft2(np.fft.fftshift(E_init)))  U(x,y;0) or A(fx,fy;0) ?
        prop_mat =self._get_prop_mat(l)
        E_K *= prop_mat
        E_out = self._my_ifft2(E_K) # np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(E_K))) # U(x,y;z)
        return E_out
    
    def reversed_free_space_propagation(self,E_init,l):
        E_K = self._my_fft2(E_init) # np.fft.fftshift(np.fft.fft2(np.fft.fftshift(E_init)))  U(x,y;0) or A(fx,fy;0) ?
        prop_mat =self._get_prop_mat(l,rev = True)
        E_K *= prop_mat
        E_out = self._my_ifft2(E_K) # np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(E_K))) # U(x,y;z)
        return E_out
    
    def propagate_thin(self,E_init):
        E_slm = E_init.copy()*np.exp(1j*self.diffuser1_phase)
        E_diffuser = E_slm*np.exp(1j*self.slm_phase)
        E_out = self._my_fft2(E_diffuser)
        return E_out
    
    def propagate_thick(self,E_init):
        E_slm = E_init.copy()*np.exp(1j*self.slm_phase)
        E_diffuser1 = E_slm.copy()*np.exp(1j*self.diffuser1_phase)
        E = self.free_space_propagation(E_diffuser1,self.thick_diffuser_width)
        E_diffuser2 = E* np.exp(1j*self.diffuser2_phase)
        E_out_fsp = self.free_space_propagation(E_diffuser2,self.L)
        E_out_fur = self._my_fft2(E_diffuser2)
        return E_out_fsp,E_out_fur
    
    def propagate_thick_with_angle(self,E_start,angle):
        E_init = E_start.copy()*np.exp(1j*angle*(self.XX+self.YY))
        E_init /= np.sqrt(np.sum(np.abs(E_init**2)))
        E_slm = E_init.copy()*np.exp(1j*self.slm_phase)
        E_diffuser1 = E_slm*np.exp(1j*self.diffuser1_phase)
        E = self.free_space_propagation(E_diffuser1,self.thick_diffuser_width)
        E_diffuser2 = E* np.exp(1j*self.diffuser2_phase)
        E_out_fsp = self.free_space_propagation(E_diffuser2,self.L)
        E_clear = E_diffuser2/np.exp(1j*angle*( self.XX+ self.YY))
        E_out_fur = self._my_fft2(E_clear)
        return E_out_fsp,E_out_fur
    
    def propagate_reverse_thick_f(self,E_init):
        E_prop = self._my_fft2(E_init)
        E_diffuser2 =E_prop*np.exp(-1j*self.diffuser2_phase)
        E = self.reversed_free_space_propagation(E_diffuser2,self.thick_diffuser_width)
        E_diffuser1 = E* np.exp(-1j*self.diffuser1_phase)
        E_slm = E_diffuser1*np.exp(1j*self.slm_phase)
        return E_slm

    def calc_one_thick_diff_mem(self,d):
        self.thick_diffuser_width =d #m
        # E_start = np.exp(-(SanityClass.XX**2+SanityClass.YY**2)/(SanityClass.sig)**2)
        E_start = np.exp(-(self.XX**2+ self.YY**2)/( self.sig/30)**2) #the sigma of the focus i want to achieve - nack prop from here to find slm phase

        E_init = E_start/np.sqrt(np.sum(np.abs(E_start**2)))
        E_p = self.propagate_reverse_thick_f(E_init)
        self.slm_phase = np.angle(E_p)
        Is = np.zeros(30)
        ind_row, ind_col = 512,512
        l=0
        r= np.pi/150
        phase_vec = np.linspace(0,r,30)
        for i,phase in enumerate(phase_vec):
            a = (2*np.pi/ Speckles.wl)*np.sin(phase)
            E_start = np.exp(-( self.XX**2+self.YY**2)/(self.sig/5)**2) #the sigma of the laser, this one is important to speckle grain
            E, E_fur = self.propagate_thick_with_angle(E_start,a)
            E_fur/= np.sqrt((np.abs(E_fur)**2).sum())
            max_speck = E_fur[ind_row-l:ind_row+l+1,ind_col-l:ind_col+l+1]
            I_ronen = ((np.abs(max_speck)**2)).sum()
            Is[i] = I_ronen
        return Is


    def snapshot(self,name):
        current_timestamp = datetime.now()
        formatted_timestamp = current_timestamp.strftime("%y_%m_%d_%H_%M_%S")
        filename = r"G:\My Drive\People\Noa\Code\code WFS\results\{}_{}".format(name,formatted_timestamp)
        np.savez(filename, slm_phase = self.slm_phase, diffuser1_phase = self.diffuser1_phase,
                 diffuser2_phase = self.diffuser2_phase)



# methods for mosk
    def get_cost(self,E_init):
        if self.is_thick == False:
            E_out = self.propagate_thin(E_init)
        if self.is_thick == True:
            E_out = self.propagate_thick(E_init)
        I = np.sum(np.abs(E_out[Speckles.roi])**2)
        return I

    def find_best_phi(self,A, i, j,phase_vec,E_init):
        Is = np.zeros(len(phase_vec))
        for ind, phase in enumerate(phase_vec):
            A[i,j] = phase
            A_scaled = cv2.resize(A, (200,200), interpolation=cv2.INTER_AREA)
            self.slm_phase[Speckles.active_slice] = A_scaled
            Is[ind] = self.get_cost(E_init)
        CC = (Is * np.exp(1j * phase_vec)).sum()
        best_phi = np.mod(np.angle(CC)+2*np.pi, 2*np.pi)
        return best_phi
