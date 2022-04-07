import numpy as np

class RBF_PoU:
    def __init__(self, data_coords, grid_coords, D, p_inf, nmax):
        self.D = D
        self.p_inf = p_inf
        self.nmax = nmax
        self.data_coords = data_coords
        self.xp, self.yp, self.zp = self.data_coords[0], self.data_coords[1], self.data_coords[2]
        self.grid_coords = grid_coords
        self.xg, self.yg, self.zg = self.grid_coords[0], self.grid_coords[1], self.grid_coords[2]

        self.c_partition = None
        self.ul_partitions = []
        self.used_partitions = []
        self.grids_used = []
        self.uppers = []
        self.lowers = []

        self.weights = []
        self.acc_weights = np.zeros(len(self.xg))
        self.norm_weights = []

    def split(self):
        x, y, z = self.xp[self.c_partition], self.yp[self.c_partition], self.zp[self.c_partition]

        #Plotting for debug
        ###
        #fig, axs = plt.subplots(1,1, figsize=(10,10))
        #_ = gs.location_plot(data, var='signed_distances_rt_1', cmap='bwr',ax=axs)

        #rect1 = Rectangle((np.min(x), np.min(y)), np.max(x)-np.min(x), np.max(y)-np.min(y), linewidth=1, edgecolor='red', facecolor='none')
        #axs.add_patch(rect1)
        ###
        
        n = len(x)
        print('Number of samples in Pc: {}'.format(n))

        if n > self.nmax:
            npart = int(np.ceil(n/2*(1+self.D)))
            print('Number of samples in Pl and Pu: {}'.format(npart))
            deltax, deltay, deltaz = np.max(x)-np.min(x), np.max(y)-np.min(y), np.max(z)-np.min(z)
            longaxis = np.argmax([deltax, deltay, deltaz])
            sortedindices = np.argsort([x, y, z][longaxis])
            lpartidx, upartidx  = sortedindices[:npart], sortedindices[n-npart:]

            self.ul_partitions.append(self.c_partition[lpartidx])
            self.ul_partitions.append(self.c_partition[upartidx])

        else:
            print('Number of samples less than the treshold')
            self.used_partitions.append(self.c_partition)

            minx, maxx = np.min(x), np.max(x)
            miny, maxy = np.min(y), np.max(y)
            minz, maxz = np.min(z), np.max(z)
            deltax, deltay, deltaz = maxx-minx, maxy-miny, maxz-minz
            percx, percy, percz = deltax*self.p_inf, deltay*self.p_inf, deltaz*self.p_inf
            minx, maxx = minx-percx, maxx+percx
            miny, maxy = miny-percy,maxy+percy
            minz, maxz = minz-percz,maxz+percz
            self.lowers.append([minx, miny, minz])
            self.uppers.append([maxx, maxy, maxz])

            fx = np.logical_and(self.xg >= minx, self.xg <= maxx)
            fy = np.logical_and(self.yg >= miny, self.yg <= maxy)
            fz = np.logical_and(self.zg >= minz, self.zg <= maxz)
            fg = np.logical_and.reduce([fx, fy, fz])
            self.grids_used.append(fg)

            self.l_partition = None
            self.u_partition = None

    def partition(self):
        if self.c_partition is None:
            self.c_partition = np.arange(len(self.xp))
            self.split()
            
        while len(self.ul_partitions) > 0:
            self.c_partition = self.ul_partitions[0]
            self.split()       
            del self.ul_partitions[0]

    def calculate_weights(self):
        for i, idxs in enumerate(self.grids_used):
            coords = np.array(self.grid_coords).T[idxs]
            wlist = []
            ux, uy, uz = self.uppers[i][0], self.uppers[i][1], self.uppers[i][2]
            lx, ly, lz = self.lowers[i][0], self.lowers[i][1], self.lowers[i][2]
            for c in coords:
                xc, yc, zc = c[0], c[1], c[2]
                #dis = 1 - ((4 * (xc-lx) * (ux-xc)) / ((ux-lx)**2) * (4 * (yc-ly) * (uy-yc)) / ((uy-ly)**2) * (4 * (zc-lz) * (uz-zc)) / ((uz-lz)**2))
                dis = 1 - ((4 * (xc-lx) * (ux-xc)) / ((ux-lx)**2) * (4 * (yc-ly) * (uy-yc)) / ((uy-ly)**2))
                wlist.append(dis)
            
            dfunc = np.array(wlist)
            wfunc = 1 - dfunc

            self.weights.append(wfunc)
            self.acc_weights[idxs] = self.acc_weights[idxs] + wfunc

    def normalize_weights(self):
        for i, idx in enumerate(self.grids_used):
            nw = self.weights[i]/self.acc_weights[idx]
            self.norm_weights.append(nw)