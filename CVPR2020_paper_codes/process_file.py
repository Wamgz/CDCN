
import os
def merge_proctocol_file_list(root_dir, file_type):
    cur_dir = [os.path.join(root_dir, 'Protocol_' + str(i)) for i in range(1, 5)]
    res = []
    for path in cur_dir:
        exist_files = os.listdir(path)
        for file in exist_files:
            if file.startswith(file_type) and file.endswith('.txt'):
                with open(os.path.join(path, file), 'r') as f:
                    lines = f.readlines()
                    print(str(len(lines)) + ' ' + file_type + ' ' + path)
                    for line in lines:
                        res.append(line)
    print(len(res))
    # res = []
    # for file in file_path:
    #     with open(file, 'r') as f:
    #         lines = f.readlines()
    #         print(str(len(lines)) + ' ' + file_type)
    #         for line in lines:
    #             res.append(line)
    # print(len(res))
if __name__ == '__main__':
    merge_proctocol_file_list('/root/autodl-tmp/oulu/Protocols', 'Train')
    # merge_proctocol_file_list('/root/autodl-tmp/oulu/Protocols', 'Test')
    # merge_proctocol_file_list('/root/autodl-tmp/oulu/Protocols', 'Dev')