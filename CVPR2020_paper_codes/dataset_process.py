import os
import numpy as np

## 获取各个mode下的avi文件路径
if __name__ == '__main__':
    root_dir = '/root/autodl-tmp/oulu'
    depth_dir = '/root/autodl-tmp/oulu_depth'
    modes = ['Train', 'Test', 'Dev']
    avi_path_modes = os.path.join(root_dir, '{}_files')
    for mode in modes:
        all_files_list = []
        with open(os.path.join(root_dir, mode + '_all.txt'), 'w') as f:
            all_files_list = f.readlines()
            all_files_name_list = [file.split('/').split('.')[0] for file in all_files_list if file.endswith('.avi')]
            print('mode: ' + mode + ' ' + all_files_name_list)
        
        avi_path = avi_path_modes.format(mode)
        avis = [os.path.join(avi_path, avi) for avi in os.listdir(avi_path) if avi.endswith('.avi')]
        np.savetxt(os.path.join(root_dir, mode + '_all.txt'), avis, delimiter=',', fmt='%s')

        print(avis)