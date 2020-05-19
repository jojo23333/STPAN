import tensorflow as tf
import numpy as np
import os

FLAGS = tf.flags.FLAGS

class DataLoader(object):
    """
        Load data for 
    """
    def __init__(self, sigma_gaussian=-1):
        self.train_csv = FLAGS.train_csv
        self.dataset_path = FLAGS.dataset
        self.frame_size = FLAGS.frame_size
        self.img_shape = [128, 128]
        self.color = FLAGS.color
        self.img_h = 720
        self.img_w = 1280

    def build_train_pipeline(self, batch_size, shuffle=False):
        """
            input pipeline to build train_dataset and validation dataset
        inputs:
            --mapf: the map function performed preparation to input data [Synthesis noise in denoise case]
            --batchsize:
            --test:
            --shuffle: whether to shuffle the input data
        outputs:
            Iterator for the required dataset
        """
        with tf.device('/CPU:0'):
            if self.dataset_path != "":
                train_imgs, val_imgs = self.get_training_list(self.frame_size)
            else:
                print("Please provide dataset path!")
                exit(1)

            self.dataset_size = len(train_imgs)
            train_dataset = tf.data.Dataset.from_tensor_slices(train_imgs)#, gt_imgs))
            val_dataset = tf.data.Dataset.from_tensor_slices(val_imgs)
            if not shuffle:
                train_dataset = train_dataset.shuffle(len(train_imgs))
            val_dataset = val_dataset.shuffle(len(val_imgs))
            train_dataset = self.process_train_data(train_dataset, batch_size=batch_size)
            val_dataset = self.process_validation_data(val_dataset, batch_size=batch_size)

            print("DATASET SIZE: ", self.dataset_size)
            print("Valiadation size: ", len(val_imgs)*4)
            return train_dataset, val_dataset

    def process_train_data(self, dataset, batch_size):
        dataset = dataset.map(self.read_train_img, num_parallel_calls=8)
        dataset = dataset.map(self.random_crop_downsize, num_parallel_calls=8)
        dataset = dataset.map(self.add_random_synthetic_noise, num_parallel_calls=8)
        # dataset = dataset.flat_map(self.add_random_synthetic_noise)
        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
        #dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(batch_size)
        return dataset

    def process_validation_data(self, dataset, batch_size):
        add_validation_loss = lambda a,b,c,d: self.add_synthetic_noise(a, b, c, d, validation=True)
        dataset = dataset.map(self.read_train_img, num_parallel_calls=8)
        dataset = dataset.flat_map(self.split_image_validation)
        dataset = dataset.map(add_validation_loss, num_parallel_calls=8)
        dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
        #dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(batch_size)
        return dataset

    def read_train_img(self, imgs):
        input_imgs = []
        for i in range(self.frame_size):
            img_string = tf.read_file(imgs[i]) 
            image = tf.image.decode_png(img_string, channels=3)
            image = tf.image.convert_image_dtype(image, tf.float32) 
            input_imgs.append(image)
        #cropped_imgs = tf.map_fn(lambda x: tf.random_crop(x ,[out_h, out_w, 3]), input_imgs)
        frames = tf.stack(input_imgs)
        frames = frames ** 2.2
        if not self.color:
            frames = tf.reduce_mean(frames, axis=-1, keepdims=True)
        return frames, imgs[self.frame_size//2]

    def random_crop_downsize(self, frames, img_str):
        out_h = self.img_shape[0]
        out_w = self.img_shape[1]

        if self.color:
            color_ch = 3
        else:
            color_ch = 1
        frames = tf.random_crop(frames, [self.frame_size, out_h, out_w, color_ch])
        # frames = tf.image.resize_images(frames, [out_h, out_w])
        frames = tf.transpose(frames, [1, 2, 0, 3])
        print("FRAMES", frames.get_shape())
        return frames, img_str

    def split_image_validation(self, frames, img_str):
        """
            split image for validation
        """
        img_h = self.img_h
        img_w = self.img_w
        out_h = self.img_shape[0]
        out_w = self.img_shape[1]

        frame1 = tf.image.crop_to_bounding_box(frames, img_h//4-out_h//2, img_w//4-out_w//2, out_h, out_w)
        frame2 = tf.image.crop_to_bounding_box(frames, img_h//4-out_h//2, img_w*3//4-out_w//2, out_h, out_w)
        frame3 = tf.image.crop_to_bounding_box(frames, img_h*3//4-out_h//2, img_w//4-out_w//2, out_h, out_w)
        frame4 = tf.image.crop_to_bounding_box(frames, img_h*3//4-out_h//2, img_w*3//4-out_w//2, out_h, out_w)
        frame1 = tf.transpose(frame1, [1, 2, 0, 3])
        frame2 = tf.transpose(frame2, [1, 2, 0, 3])
        frame3 = tf.transpose(frame3, [1, 2, 0, 3])
        frame4 = tf.transpose(frame4, [1, 2, 0, 3])

        sig_read = [0.05, 0.03, 0.02, 0.01]
        sig_shot = [0.15, 0.1,  0.08, 0.05]
        return tf.data.Dataset.from_tensor_slices(([frame1, frame2, frame3, frame4], [img_str]*4,  sig_read, sig_shot))
    
    def add_synthetic_noise(self, frames, img_str, sig_read, sig_shot, validation=False):
        """
            add random read & shot noise based on the input signal
        """
        print(frames.get_shape())
        print(sig_read.get_shape())
        ref_frame = self.frame_size // 2
        out_h = self.img_shape[0]
        out_w = self.img_shape[1]
        if validation:
            white_level = tf.ones(dtype=tf.float32, shape=[1,1,1,1])
        else:
            white_level = tf.pow(10., tf.random_uniform([1, 1, 1, 1], np.log10(.1), np.log10(1.)))
        
        frames = frames * white_level
        gt = frames[...,ref_frame,:]

        # add noise for each frame, ps: noise level is same for income imgs
        read_noise = sig_read * tf.random_normal(tf.shape(frames))
        shot_noise = sig_shot * tf.sqrt(frames) * tf.random_normal(tf.shape(frames))
        noise_frames = frames + read_noise + shot_noise
        noise_frames = tf.clip_by_value(noise_frames, 0.0, 1.0)

        sig_read = sig_read * tf.ones(shape=[out_h, out_w, 1, 1], dtype=tf.float32)
        sig_shot = sig_shot * tf.ones(shape=[out_h, out_w, 1, 1], dtype=tf.float32)
        noise_level = tf.concat([sig_read, sig_shot], axis=-1)
        noise_level = tf.reshape(noise_level, [out_h, out_w, 2])

        return noise_frames, gt, img_str, noise_level, white_level

    def add_random_synthetic_noise(self, frames, img_str):
        """
            add random read & shot noise
            read_signal in range: [1e-2, 1e-4]
            shot_signal in range: [1e-1.5, 1e-3]
        """
        ref_frame = self.frame_size // 2
        out_h = self.img_shape[0]
        out_w = self.img_shape[1]
        white_level = tf.pow(10., tf.random_uniform([1, 1, 1, 1], np.log10(.1), np.log10(1.)))
        
        frames = frames * white_level
        gt = frames[...,ref_frame,:]

        # add noise for each frame, ps: noise level is same for income imgs
        sig_read = tf.pow(10., tf.random_uniform([1, 1, 1, 1], -3., -1.5))
        sig_shot = tf.pow(10., tf.random_uniform([1, 1, 1, 1], -2., -1.))
        read_noise = sig_read * tf.random_normal(tf.shape(frames))
        shot_noise = sig_shot * tf.sqrt(frames) * tf.random_normal(tf.shape(frames))
        noise_frames = frames + read_noise + shot_noise
        noise_frames = tf.clip_by_value(noise_frames, 0.0, 1.0)

        sig_read = tf.tile(sig_read, [out_h, out_w, 1, 1])
        sig_shot = tf.tile(sig_shot, [out_h, out_w, 1, 1])
        noise_level = tf.concat([sig_read, sig_shot], axis=-1)
        noise_level = tf.reshape(noise_level, [out_h, out_w, 2])

        return noise_frames, gt, img_str, noise_level, white_level

        # noise_frames_2 = sig_read * tf.random_normal(tf.shape(frames)) + sig_shot * tf.sqrt(frames) * tf.random_normal(tf.shape(frames)) + frames


        # return tf.data.Dataset.from_tensor_slices(([noise_frames, noise_frames_2], [gt]*2, [img_str]*2, [noise_level]*2, [white_level]*2))


    def load_noised_img(self, imgs, img_shape, frame_size, degamma=2.2, COLOR=True, TEST=False):
        """
            Process input data.Performing jitter -> gamma correction -> linear scale -> noise
        """
        out_h = img_shape[0]
        out_w = img_shape[1]
        reference_frame = frame_size // 2
        input_imgs = []
        # crop proccess
        for i in range(frame_size):
            img_string = tf.read_file(imgs[i]) 
            image = tf.image.decode_png(img_string, channels=3)
            image = tf.image.convert_image_dtype(image, tf.float32) 
            input_imgs.append(image)
        #cropped_imgs = tf.map_fn(lambda x: tf.random_crop(x ,[out_h, out_w, 3]), input_imgs)
        frames = tf.stack(input_imgs)
        color_ch = 3

        frames = tf.random_crop(frames, [frame_size, out_h*2, out_w*2, color_ch])
        frames = tf.image.resize_images(frames, [out_h, out_w])
        frames = frames ** degamma
        if not COLOR:
            frames = tf.reduce_mean(frames, axis=-1, keepdims=True)
            color_ch = 1
        frames = tf.transpose(frames, [1, 2, 0, 3])
        print("FRAMES", frames.get_shape())

        white_level = tf.pow(10., tf.random_uniform([1, 1, 1, 1], np.log10(.5), np.log10(1.)))
        frames = frames * white_level

        gt = frames[...,reference_frame,:]
        # add noise for each frame, ps: noise level is same for income imgs
        sig_read = tf.pow(10., tf.random_uniform([1, 1, 1, 1], -3., -1.5))
        sig_shot = tf.pow(10., tf.random_uniform([1, 1, 1, 1], -2., -1.))
        read_noise = sig_read * tf.random_normal(tf.shape(frames))
        shot_noise = sig_shot * tf.sqrt(frames) * tf.random_normal(tf.shape(frames))
        noise_frames = frames + read_noise + shot_noise
        noise_frames = tf.clip_by_value(noise_frames, 0.0, 1.0)

        sig_read = tf.tile(sig_read, [out_h, out_w, 1, 1])
        sig_shot = tf.tile(sig_shot, [out_h, out_w, 1, 1])
        noise_level = tf.concat([sig_read, sig_shot], axis=-1)
        noise_level = tf.reshape(noise_level, [out_h, out_w, 2])

        return noise_frames, gt, imgs[reference_frame], noise_level, white_level

    def get_training_list(self, T):
        """
            Find the training dataset according to the dataset layout
        """
        self.videos = {}
        videos = [x for x in os.listdir(self.dataset_path) if os.path.isdir(os.path.join(self.dataset_path, x))]
        
        # list out a tree_dict of dataset
        for v in videos:
            video_path = os.path.join(self.dataset_path, v)
            scene_list = [x for x in os.listdir(video_path) if os.path.isdir(os.path.join(video_path, x))]
            scenes = {}
            for s in scene_list:
                scene_path = os.path.join(video_path, s, "gt")
                frame_list = [os.path.join(scene_path, x) for x in os.listdir(scene_path) if x.endswith("png")]
                frame_list.sort()
                
                scenes[s] = frame_list
            self.videos[v] = scenes

        # add all five frame into train_list
        train_images = []
        val_images = []
        val_cnt = 0
        for v, sc in self.videos.items():
            print("video: ", v)
            for s, f in sc.items():
                print("  scenes: ", s, ", ", len(f))
                train_list = [np.array(f[i:i+T]) for i in range(0, len(f)-T)]
                for p in train_list:
                    if val_cnt % 20 == 1:
                        val_images.append(p)
                    else:
                        train_images.append(p)
                    val_cnt += 1

        train_images = np.array(train_images)
        val_images = np.array(val_images)
        print(train_images.shape)
        return train_images, val_images
