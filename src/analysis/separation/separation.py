import numpy as np
import annoy
import pickle as pkl


"""
sample two points from the dataset, in order to evaluate density along the connecting line
"""
def sample_points(array):
    return np.random.choice(array,2,replace=False)

"""
build index from dataset for nearest neighbour requests
"""
def build_index(dataset,n_trees=500,on_disk=True,path="index.ann"):
    ind = annoy.AnnoyIndex(dataset.shape[1],"euclidean")
    for i,d in enumerate(dataset):
        ind.add_item(i,d)
    if on_disk:
        ind.on_disk_build(path)
    ind.build(n_trees)
    return ind

"""
get biggest radius around x1,x2, s.t. for one of the points, the ball of that radius around the point contains only k other points
"""
def get_radius(ind,x1,x2,k=5):
    nns1 = ind.get_nns_by_vector(x1,k+2)
    nns2 = ind.get_nns_by_vector(x2,k+2)
    prev_max_dist = 0
    max_dist = 0
    for nn in nns1:
        dist = np.linalg.norm(x1-ind.get_item_vector(nn),ord=2)
        if dist>prev_max_dist:
            prev_max_dist = min(dist,max_dist)
            max_dist = max(dist,max_dist)
    radius1 = .5*(prev_max_dist+max_dist)

    prev_max_dist = 0
    max_dist = 0
    for nn in nns2:
        dist = np.linalg.norm(x2-ind.get_item_vector(nn),ord=2)
        if dist>prev_max_dist:
            prev_max_dist = min(dist,max_dist)
            max_dist = max(dist,max_dist)
    radius2 = .5*(prev_max_dist+max_dist)

    return max(radius1,radius2)

"""
search for points of distance <= delta to the connecting line, calculate intersections of delta-balls with the line
"""
def get_points_intervalls_cylinder(ind,x1,x2,delta,use_index=True,iter=10000, max_points=1E5):
    center = .5*(x1+x2)
    g_length = np.linalg.norm(x2-x1,ord=2)
    radius = .5*g_length+delta

    n = iter
    max_dist = 0
    nns = []
    old_nns = []

    if use_index:
        while n <= max_points:
            print(n)
            old_nns = nns
            nns = ind.get_nns_by_vector(center,n)
            passed = False
            for i,nn in enumerate(nns):
                if not nn in old_nns:
                    dist = np.linalg.norm(center-ind.get_item_vector(nn),ord=2)
                    max_dist = max(dist,max_dist)
                    if max_dist >= radius:
                        passed = True
                        break
            if passed:
                break
            n += iter

        ball_points = [ind.get_item_vector(nn) for nn in nns]
        intervalls = []
    else:
        ball_points = ind
        intervalls = []


    def orthogonal_projection(point):
        g = (x2-x1)/g_length
        p = (point-x1)/g_length
        return x1+np.inner(g,p)*(x2-x1)

    for i,point in enumerate(ball_points):
        proj_p = orthogonal_projection(point)
        dist_g = np.linalg.norm(point-proj_p,ord=2)
        if dist_g < delta:
            intervall_center_relative = np.linalg.norm(x1-proj_p,ord=2)/g_length
            range_relative = (delta**2-dist_g**2)**.5/g_length
            intervall = (intervall_center_relative-range_relative,intervall_center_relative+range_relative)
            if not intervall[0]>=1 or intervall[1]<=0:
                intervall = (max(0,intervall[0]),min(1,intervall[1])) # chop off parts outside of g
                intervalls.append((i,intervall))

    return ball_points,intervalls


def get_segment_counts(intervalls,include_points=False):
    boundaries = list(set([0,1] + [a for _,(a,b) in intervalls] + [b for _,(a,b) in intervalls]))
    counts = []
    points = []
    boundaries.sort()
    for i in range(len(boundaries)-1):
        segment_count = 0
        segment_points = []
        left = boundaries[i]
        right = boundaries[i+1]

        for j,(a,b) in intervalls:
            if a<=left and b>=right:
                segment_count += 1
                if include_points:
                    segment_points.append(j)
        counts.append(segment_count)
        if include_points:
            points.append(segment_points)
    if include_points:
        return boundaries,counts,points
    return boundaries,counts
