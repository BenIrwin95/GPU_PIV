import h5py
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
img = mpimg.imread('../cam1_im_000_A.tiff')

f = h5py.File('../output.h5', 'r')
last_pass_idx = f.attrs["N_pass"]-1
last_frame_idx = f.attrs["N_frames"]-1
group = "Pass_" + str(last_pass_idx)
X = f[group]["X"][:]
Y = f[group]["Y"][:]
U = f[group]["U"]["frame{:03d}".format(last_frame_idx)][:]
V = f[group]["V"]["frame{:03d}".format(last_frame_idx)][:]


step=1;
plt.figure(figsize=(6, 6))
plt.imshow(img, extent=[X.min(), X.max(), Y.min(), Y.max()], cmap='gray')
plt.quiver(X[::step, ::step], Y[::step, ::step], U[::step, ::step], V[::step, ::step],units='dots',       # Arrow dimensions in dots (pixels)
                scale_units='dots', # Arrow length scaling in dots (pixels)
                scale=0.9,            # A reference scale (e.g., 1 data unit = 1 pixel)
                width=2,            # Shaft width in 'dots' (e.g., 2 pixels wide)
                headwidth=2,        # Head width as multiple of shaft width
                headlength=2,       # Head length as multiple of shaft width
                headaxislength=2,   # Head length at shaft intersection)
                color='red')

#plt.grid(True)
plt.axis('equal')
plt.show()

#
# plt.figure(figsize=(6, 6))
# slc = 100
# plt.plot(Y[:,slc], U[:,slc])
# plt.show()
#
