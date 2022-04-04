from scipy.spatial import distance
import numpy as np

def min_dist(point, points):
    dist_matrix = distance.cdist([point], points, "euclidean")
    return np.amin(dist_matrix)

def sdcalc(data, x, y, z, cat):

    x = data[x].values
    y = data[y].values
    if z == None:
        z = np.ones(len(x)) * 0.5
    else:
        z = data[z]
    
    prop_values = data[cat].values

    coords_matrix = np.vstack((x,y,z)).T

    #calculating signed distances
    nan_filter = np.isfinite(prop_values)
    unique_rts = np.unique(prop_values[nan_filter])

    for rt in unique_rts:
        print('calculating signed distances for rock type {}'.format(int(rt)))
        filter_0 = prop_values != rt
        filter_1 = prop_values == rt
        points_0 = coords_matrix[filter_0]
        points_1 = coords_matrix[filter_1]

        sd_prop = []
        for idx, pt in enumerate(prop_values):
            if np.isnan(pt):
                sd_prop.append(float('nan'))

            else:
                point = coords_matrix[idx]
                if pt == rt:
                    sd_prop.append(-min_dist(point, points_0))
                else:
                    sd_prop.append(min_dist(point, points_1)) 
    
        data['signed_distances_rt_{}'.format(int(rt))] = sd_prop
