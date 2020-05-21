import os
from glob import glob

def find_subdir_leaves(dirpath):
    subdirs = [x for x in os.listdir(dirpath) if os.path.isdir(x)]
    if len(subdirs) == 0:
        return [dirpath]
    else:
        video_clips = []
        for subdir in subdirs:
            video_clips = video_clips + find_subdir_leaves(os.path.join(dirpath, subdir))
        return video_clips

def get_vimeo20k_instances(cfg, split, ext=".png"):
    """
        return a list of frame sequences
    """
    FRAME_SIZE = cfg.STPAN.FRAME_SIZE
    if split == 'train':
        dirpath = cfg.DATA.PATH_TO_TRAINING_SET
        all_sequences = []
        videos_clips = find_subdir_leaves(dirpath)
        for path_2_video in videos_clips:
            frames = glob.glob(path_2_video + '/*' + ext)
            frames.sort()
            for i in range(0, len(frames)-FRAME_SIZE):
                all_sequences.append(frames[i: i+FRAME_SIZE])
        return all_sequences
    else:
        if split == 'val':
            dirpath = cfg.DATA.PATH_TO_VAL_SET
        elif split == 'test':
            dirpath = cfg.DATA.PATH_TO_TEST_SET
        all_sequences = []
        videos_clips = find_subdir_leaves(dirpath)
        for path_2_video in videos_clips:
            noisy_frames = glob.glob(path_2_video + '/noisy*' + ext)
            gt_frames = glob.glob(path_2_video + '/gt*' + ext)
            all_sequences.append((noisy_frames, gt_frames))
        return all_sequences