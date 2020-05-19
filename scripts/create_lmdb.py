"""Create lmdb files for [General images (291 images/DIV2K) | Vimeo90K | REDS] training datasets"""

import sys
import os
import glob
import pickle
import numpy as np
import lmdb
import cv2

from tqdm import tqdm
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    dataset = 'ntire2020'  # vimeo90K | REDS | general (e.g., DIV2K, 291) | DIV2K_demo |test
    mode = 'source'  # used for vimeo90k and REDS datasets
    # btire2020: 
    # REDS: train_sharp, train_sharp_bicubic, train_blur_bicubic, train_blur, train_blur_comp
    #       train_sharp_flowx4
    if dataset == 'ntire2020':
        print("generate source lmdb")
        ntire2020('source')
        print("generate target lmdb")
        ntire2020('target')


def ntire2020(mode):
    """Create lmdb for NTIRE2020 Video quality mapping dataset
    source: [1920, 1080, 3]
    target: [1920, 1080, 3]
    """
    # Lmdb commits after BATCH imgs
    BATCH = 5000 
    H_dst = 1080
    W_dst = 1920
    if mode == "target":
        img_folder = '/Disk2/limuchen/NTIRE2020/train/target'
        lmdb_save_path = '/Disk2/limuchen/NTIRE2020/train/target_lmdb.lmdb'
    elif mode == "source":
        img_folder = '/Disk2/limuchen/NTIRE2020/train/source'
        lmdb_save_path = '/Disk2/limuchen/NTIRE2020/train/source_lmdb.lmdb'
    elif mode == "test":
        pass
    #################################################################
    if os.path.exists(lmdb_save_path):
        print('Folder [{:s}] already exists. Exit...'.format(lmdb_save_path))
        sys.exit(1)

    all_img_list = []
    keys = []
    for vid in os.listdir(img_folder):
        # TODO: ['008', '029', '034', '035'] have different shape, fix this later
        if vid in ['008', '029', '034', '035']:
            continue
        imgs = os.listdir(os.path.join(img_folder, vid))
        imgs.sort()
        for id, img in enumerate(imgs):
            all_img_list.append(os.path.join(img_folder, vid, img))
            keys.append(os.path.join(vid, img))

    all_img_list.sort()
    keys.sort()

    #### write data to lmdb
    # Asumming all imgs with the same size here
    data_size_per_img = cv2.imread(all_img_list[0], cv2.IMREAD_UNCHANGED).nbytes
    print('data size per img is: ', data_size_per_img)
    data_size = data_size_per_img * len(all_img_list)
    env = lmdb.open(lmdb_save_path, map_size=data_size * 10)
    txn = env.begin(write=True)

    for idx, (path, key) in tqdm(enumerate(zip(all_img_list, keys))):
        key_byte = key.encode('ascii')
        data = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        
        H, W, C = data.shape
        assert H == H_dst and W == W_dst and C == 3, 'Different shape in img: {}'.format(path)
        txn.put(key_byte, data)
        if idx % BATCH == BATCH -1:
            txn.commit()
            txn = env.begin(write=True)
    txn.commit()
    env.close()
    print("finish writing lmdb")

    #### create meta information only for source mode
    # currently unecessary
    meta_info = {}
    if mode == "source":
        pass
    elif mode == "test":
        pass
    return

def test_lmdb(dataroot, dataset='REDS'):
    env = lmdb.open(dataroot, readonly=True, lock=False, readahead=False, meminit=False)
    meta_info = pickle.load(open(os.path.join(dataroot, 'meta_info.pkl'), "rb"))
    print('Name: ', meta_info['name'])
    print('Resolution: ', meta_info['resolution'])
    print('# keys: ', len(meta_info['keys']))
    # read one image
    if dataset == 'ntire2020':
        key = '00001_0001_4'
    else:
        key = '000_00000000'
    print('Reading {} for test.'.format(key))
    with env.begin(write=False) as txn:
        buf = txn.get(key.encode('ascii'))
    img_flat = np.frombuffer(buf, dtype=np.uint8)
    C, H, W = [int(s) for s in meta_info['resolution'].split('_')]
    img = img_flat.reshape(H, W, C)
    cv2.imwrite('test.png', img)


if __name__ == "__main__":
    main()
