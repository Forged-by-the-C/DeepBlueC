import os

def construct_dir_path(project_dir="NepalEarthquakes",
        sub_dir="Data"):
    cwd_path = os.getcwd().split('/')
    path_to_file = []
    for d in cwd_path:
        path_to_file.append(d)
        if d == 'NepalEarthquakes':
            break
    path_to_file.append(sub_dir)
    return '/' + os.path.join(*path_to_file) + '/'

if __name__=='__main__':
    pass
