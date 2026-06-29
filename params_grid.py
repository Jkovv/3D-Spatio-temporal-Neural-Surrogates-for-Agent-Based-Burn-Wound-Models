L = 50 # lattice edge length (voxels); change to 100, 250, or 500 for larger variants
scale_factor = L / 500

nx = L
ny = L
nz = L # isotropic 3D lattice
dx = 1.
dy = 1.
dz = 1.

total_cytokines = 6
total_celltypes = 10

relaxationmcs = 10000  # CC3D steps per biological update
fipy_duration = int(1) # FiPy solve duration per call (hours)

s_mcs = 60. # seconds per MCS
h_mcs = 1 / 60. # hours per MCS

# physical domain
true_mass = 1
true_size = 5  # 5 cm physical domain

lineconv = true_size / L # cm per voxel
volumeconv = (true_size / L) ** 3 # cm³ per voxel
massconv = true_mass