from logging import raiseExceptions
import os.path as osp
import random

def load_annotations(data_root, split):
        
        invalid_depth_num = 0
        img_infos = []
        if split is not None:
            with open(split) as f:
                for line in f:
                    img_info = dict()
                    depth_map = line.strip().split(",")[1]
                    depth_map = depth_map.replace('/', '\\')
                    if depth_map == 'None':
                        invalid_depth_num += 1
                        continue
                    img_info['depth_map'] = osp.join(data_root, remove_leading_slash(depth_map))
                    img_name = line.strip().split(",")[0]
                    img_name = img_name.replace('/', '\\')
                    img_info['filename'] = osp.join(data_root, remove_leading_slash(img_name))
                    img_infos.append(img_info)
        else:
            raise NotImplementedError 

        # github issue:: make sure the same order
        img_infos = sorted(img_infos, key=lambda x: x['filename'])
        print(f'Loaded {len(img_infos)} images.')
        print(f'Invalid {invalid_depth_num}')
        return img_infos

def remove_leading_slash(s):
    if s[0] == '/' or s[0] == '\\':
        return s[1:]
    return s


img_infos = load_annotations('C:\\Users\\psiml8\\Documents\\GitHub\\PSIML', "C:\\Users\\psiml8\\Documents\\GitHub\\PSIML\\data\\nyu2_train.csv")
random.shuffle(img_infos)
num_rows = len(img_infos)
train_infos = img_infos[:5000]
val_infos = img_infos[5000:6000]

"""from PIL import Image
image = Image.open(img_infos[0]["depth_map"])
image.show()
image1 = Image.open(img_infos[0]["filename"])
image1.show()
"""
