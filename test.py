import os, time, argparse
import random, sys, cv2
import tensorflow as tf
import numpy as np
import pandas as pd

from datetime import datetime
from PIL import Image

from stpan import stpan

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer('batch_size', 1, 'Number of images in a single batch')
tf.flags.DEFINE_integer('input_size', 128, 'Height and width for input image patches for train')
tf.flags.DEFINE_integer('max_offset', 128, 'Default offset is obtained by multiply max_offset after normlized.')
tf.flags.DEFINE_integer('frame_size', 5, 'Input number of frame')
tf.flags.DEFINE_integer('kernel_size_t', 3, 'Default kernel size on the temporal dimension. T*K*K')
tf.flags.DEFINE_integer('kernel_dilate_rate', 1, 'Set this term > 1 if you want to use dilated kernel.')
tf.flags.DEFINE_integer('kernel_size_k', 3 , 'Default kernel size on the spatial dimension. T*K*K')
tf.flags.DEFINE_bool('color', False, 'Whether to denoise on colored image.')

tf.flags.DEFINE_string('noise_level_type', 'single_std', 'Noise level type input into the network')
tf.flags.DEFINE_string('test_csv', '/data/DATASET/STTN_SCENE/test_Scene_1/test_data_Scene_5_1.csv', 'Csv file recording test info. DEPRECATED.')
tf.flags.DEFINE_string('test_dir', '', 'Path to test dir')
tf.flags.DEFINE_string('output_root', '/data/EXP_RESULTS/', 'Test output.')
tf.flags.DEFINE_string('model_path', '', 'Input model path')
tf.flags.DEFINE_string('name', 'Scene_L1', 'Output name')
tf.flags.DEFINE_integer('save_cycle', -1, 'The cycle to save optical flow and weight information, -1 stands for not saving anything')
tf.flags.DEFINE_float('sig_read', 0.02, 'sig read')
tf.flags.DEFINE_float('sig_shot', 0.08, 'sig shot')

TEST_W = 512
TEST_H = 288
STEP_SIZE_W = 256
STEP_SIZE_H = 144

DEGAMMA = 2.2
IMAGE_W = 1280
IMAGE_H = 720
FRAME_SIZE = 5

w_split_num = (IMAGE_W-TEST_W-1) // STEP_SIZE_W + 2
h_split_num = (IMAGE_H-TEST_H-1) // STEP_SIZE_H + 2
SPLIT_NUM = w_split_num * h_split_num
print("SPLIT NUM: ", SPLIT_NUM)
print("W_SPLIT_NUM, H_SPLIT_NUM: %d %d" % (w_split_num, h_split_num))
print("STEP SIZE: %d" % SPLIT_NUM)

def load_test_img(imgs, gt_str, sig_read, sig_shot, frame_size, degamma=2.2, COLOR=False):
    reference_frame = frame_size // 2
    input_imgs = []
    for i in range(frame_size):
        img_string = tf.read_file(imgs[i])
        image = tf.image.decode_png(img_string, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize_images(image, [IMAGE_H, IMAGE_W])
        input_imgs.append(image)
    gt = tf.read_file(gt_str)
    gt = tf.image.decode_png(gt, channels=3)
    gt = tf.image.convert_image_dtype(gt, tf.float32)
    gt = tf.image.resize_images(gt, [IMAGE_H, IMAGE_W])

    noisy = tf.stack(input_imgs, axis=-2)
    noisy = noisy ** degamma
    gt = gt ** degamma
    if not COLOR:
        noisy = tf.reduce_mean(noisy, axis=-1, keepdims=True)
        gt = tf.reduce_mean(gt, axis=-1, keepdims=True)
    print("FRAMES", noisy.get_shape())

    global TEST_H, TEST_W
    if COLOR:
        TEST_H = IMAGE_H
        TEST_W = IMAGE_W
    sig_read = sig_read * tf.ones([TEST_H, TEST_W, 1, 1])
    sig_shot = sig_shot * tf.ones([TEST_H, TEST_W, 1, 1])
    noise_level = tf.concat([sig_read, sig_shot], axis=-1)
    noise_level = tf.reshape(noise_level, [TEST_H, TEST_W, 2])

    return noisy, gt, gt_str, noise_level

def split_image(noisy, gt, gt_str, noise_level):
    noisy_list = []
    gt_list = []
    noisy = tf.transpose(noisy, [2, 0, 1, 3])
    for j in range(h_split_num):
        for i in range(w_split_num):
            y = j * STEP_SIZE_H
            x = i * STEP_SIZE_W
            print(y,x,TEST_H,TEST_W)
            noisy_cropped = tf.image.crop_to_bounding_box(noisy, y, x, TEST_H, TEST_W)
            noisy_list.append(tf.transpose(noisy_cropped, [1, 2, 0, 3]))
            gt_list.append(tf.image.crop_to_bounding_box(gt, y, x, TEST_H, TEST_W))
    str_list = [gt_str] * SPLIT_NUM
    noise_level_list = [noise_level] * SPLIT_NUM
    return tf.data.Dataset.from_tensor_slices((noisy_list, gt_list, str_list, noise_level_list))

def merge_image(img_list):
    GAMMA = 1.0 / DEGAMMA
    mask = np.zeros([IMAGE_H, IMAGE_W])
    out  = np.zeros([IMAGE_H, IMAGE_W])
    for j in range(h_split_num):
        for i in range(w_split_num):
            y = j * STEP_SIZE_H
            x = i * STEP_SIZE_W
            mask[y:y+TEST_H, x:x+TEST_W] += 1.0
            out[y:y+TEST_H, x:x+TEST_W] += img_list[j*w_split_num+i] ** GAMMA
    out = (255.0 * out) / mask
    return out.astype(np.uint8)

class TestSolver(object):
    def __init__(self):
        """
        Some setting up for the output dirs
        """
        self.test_dir = FLAGS.test_dir
        self.test_csv = FLAGS.test_csv
        self.name = FLAGS.name
        self.sig_read = FLAGS.sig_read
        self.sig_shot = FLAGS.sig_shot
        print("Noise signals", self.sig_read, self.sig_shot)
        if not os.path.exists("runs"):
            os.mkdir("runs")
        self.run_root = os.path.join(FLAGS.output_root, FLAGS.model_path.split('/')[-3])
        self.test_output = os.path.join(self.run_root, "out", FLAGS.name)
        if not os.path.exists(self.test_output):
            os.makedirs(self.test_output)

    def get_test_list(self, test_dir):
        self.test_set = []
        self.gt_set = []
        scenes = os.listdir(test_dir)
        for sc in scenes:
            burst_list = os.listdir(os.path.join(test_dir, sc))
            for img_id in burst_list:
                img_path = os.path.join(test_dir, sc, img_id)
                gts = [os.path.join(img_path, x) for x in os.listdir(img_path) if x.startswith("gt")]
                gts.sort()
                noisy = [os.path.join(img_path, x) for x in os.listdir(img_path) if x.startswith("noisy")]
                noisy.sort()
                self.test_set.append(noisy)
                self.gt_set.append(gts[FRAME_SIZE//2])
        test_size = len(self.test_set)
        self.test_set = np.array(self.test_set)
        self.gt_set = np.array(self.gt_set)
        print("Test num: ", test_size)
        return test_size


    def build_test_input_pipeline(self, mapf, batch_size, test_shuffle=False, split=True):
        if self.test_dir:
            test_size = self.get_test_list(self.test_dir)
            sig_read = np.array([self.sig_read] * test_size, dtype=np.float32)
            sig_shot = np.array([self.sig_shot] * test_size, dtype=np.float32)
            dataset = tf.data.Dataset.from_tensor_slices((self.test_set, self.gt_set, sig_read, sig_shot))
        else:
            imglist = pd.read_csv(self.test_csv)
            imgs = np.array(imglist["noisy"])
            gts = np.array(imglist["gt"])
            sig_read = np.array(imglist["sig_read"], dtype=np.float32)
            sig_shot = np.array(imglist["sig_shot"], dtype=np.float32)
            imgs = np.array([x.split("  ") for x in imgs])
            dataset = tf.data.Dataset.from_tensor_slices((imgs, gts, sig_read, sig_shot))
        if test_shuffle:
            dataset = dataset.shuffle(test_size)
        dataset = dataset.map(mapf, num_parallel_calls=8)
        if split:
            dataset = dataset.flat_map(split_image)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(1)
        iterator = dataset.make_initializable_iterator()
        return iterator

    def test(self, ckpt_path):
        """
            test proccess
        """
        BATCH_SIZE = FLAGS.batch_size
        INPUT_H = INPUT_W = FLAGS.input_size
        FRAME_SIZE = FLAGS.frame_size
        COLOR = FLAGS.color
        COLOR_CH = 3 if COLOR else 1

        MODEL_PATH = FLAGS.model_path
        NOISE_LEVEL_TYPE = FLAGS.noise_level_type
        CYCLE = FLAGS.save_cycle

        noise_function = lambda a, b, c, d: load_test_img(a, b, c, d, frame_size=FRAME_SIZE, degamma=DEGAMMA,
                                                          COLOR=COLOR)
        data_iterator = self.build_test_input_pipeline(mapf=noise_function, batch_size=1)#, test_shuffle=True)
        next_element = data_iterator.get_next()
        test_input, test_gt, img_name, noise_level = next_element
        sig_read_single_std = tf.sqrt(noise_level[..., 0:1] ** 2 + tf.maximum(0., test_input[..., FRAME_SIZE // 2, :]) * noise_level[...,1:2] ** 2)
        noise_level = noise_level if NOISE_LEVEL_TYPE == "double" else sig_read_single_std if NOISE_LEVEL_TYPE == "single_std" else None

        model = stpan

        GAMMA = 1.0 / DEGAMMA
        # shared_model = tf.make_template('shared_model', df3c_only)
        x, op_flow, weights, per_point_out = model(test_input, noise_level, NOISE_LEVEL_TYPE,
                                                            C=COLOR_CH, batch_size=1)

        # x = tf.where(tf.is_nan(x), tf.zeros_like(x), x)
        x = tf.clip_by_value(x, .0, 1.)
        x_out = tf.image.convert_image_dtype(x ** GAMMA, tf.uint8, True)
        gt_out = tf.image.convert_image_dtype(test_gt ** GAMMA, tf.uint8, True)
        yt_in = tf.image.convert_image_dtype(test_input ** GAMMA, tf.uint8, True)

        gt_MSE = tf.losses.mean_squared_error(x ** GAMMA, test_gt ** GAMMA)
        base_MSE = tf.losses.mean_squared_error(test_input[:, :, :, FRAME_SIZE // 2, :] ** GAMMA, test_gt ** GAMMA)
        gt_ssim = tf.image.ssim(x ** GAMMA, test_gt ** GAMMA, max_val=1.0)
        base_ssim = tf.image.ssim(test_input[:, :, :, FRAME_SIZE // 2, :] ** GAMMA, test_gt ** GAMMA, max_val=1.0)

        average = tf.reduce_mean(test_input, axis=-2)
        average_mse = tf.losses.mean_squared_error(average ** GAMMA, test_gt ** GAMMA)
        average_ssim = tf.image.ssim(average ** GAMMA, test_gt ** GAMMA, max_val=1.0)

        epoch_num = ckpt_path.split('/')[-1].split('.')[0]
        out = open(os.path.join(self.run_root, "output_%s_%s.csv" % (epoch_num, self.name)), "w")
        out.write("img_name,mse,psnr,base_mse,ssim,base_ssim,average_psnr,average_ssim\n")
        saver = tf.train.Saver()
        psnr_all = []
        ssim_all = []
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, ckpt_path)
            sess.run(data_iterator.initializer)
            cnt = 0
            run_options = tf.RunOptions(report_tensor_allocations_upon_oom=True)
            flow_spatial_list = []
            flow_temporal_list = []
            while True:
                try:
                    x_list = []
                    mse_list = []
                    base_mse_list = []
                    ssim_list = []
                    b_ssim_list = []
                    a_mse_list = []
                    a_ssim_list = []
                    for i in range(SPLIT_NUM):
                        stime = time.time()
                        x_, next_, output, gt_output, flow_output, tr_input, mse, b_mse, img_str, w, ssim, b_ssim ,a_mse , a_ssim= \
                            sess.run([x, next_element, x_out, gt_out, op_flow, yt_in, gt_MSE, base_MSE, img_name, weights, gt_ssim, base_ssim, average_mse, average_ssim]
                                            , options=run_options)
                        etime = time.time()

                        img_str = img_str[0].decode('utf-8').rsplit('.', 1)[0]
                        video = "/".join(img_str.split('/')[-4:-2])
                        frame = img_str.split('/')[-1].split('_')[-1]
                        if not os.path.exists(os.path.join(self.test_output, video)):
                            # print("Making dir!")
                            os.makedirs(os.path.join(self.test_output, video))
                        print(os.path.join(self.test_output, video, frame + "_out_%d.png" % i), end='\r')
                        tr_input_ref = np.reshape(tr_input[0, :, :, :, :], [TEST_H, TEST_W, FRAME_SIZE, COLOR_CH])
                        
                        flow_spatial_list.append(np.mean(flow_output[..., 2]))
                        flow_temporal_list.append(np.mean(flow_output[..., :2]))

                        path = os.path.join(self.test_output, video, frame, str(i))

                        if CYCLE != -1 and cnt % CYCLE == 0 and (i==10 or i == 6):
                            if not os.path.exists(path):
                                os.makedirs(path)
                            np.save(os.path.join(path, "flow.npy" ), flow_output[0,...])
                            np.save(os.path.join(path, "flow_weights.npy"), w[0,...,0])
                            for k in range(FRAME_SIZE):
                                cv2.imwrite(os.path.join(path, "%d_in.png" % k), tr_input_ref[..., k, 0])
                            cv2.imwrite(os.path.join(path,"out.png"), output[0,..., 0])

                        x_list.append(x_[0,...,0])
                        mse_list.append(mse)
                        base_mse_list.append(b_mse)
                        ssim_list.append(ssim)
                        b_ssim_list.append(b_ssim)
                        a_mse_list.append(a_mse)
                        a_ssim_list.append(a_ssim)

                    path = os.path.join(self.test_output, video, frame)
                    img_merged = merge_image(x_list)
                    cv2.imwrite(path+"_out.png", img_merged)

                    mse = np.mean(mse_list)
                    b_mse = np.mean(base_mse_list)
                    ssim = np.mean(ssim_list)
                    b_ssim = np.mean(b_ssim_list)
                    a_mse = np.mean(a_mse_list)
                    a_ssim = np.mean(a_ssim_list)
                    psnr = 10 * np.log10(pow(1., 2) / mse)
                    a_psnr = 10 * np.log10(pow(1., 2) / a_mse)
                    msg = "\"%s.png\",%f,%f,%f,%f,%f,%f,%f\n" % (img_str, mse, psnr, b_mse, ssim, b_ssim, a_psnr, a_ssim)
                    print(msg, end="")
                    out.write(msg)
                    cnt += 1
                    psnr_all.append(psnr)
                    ssim_all.append(ssim)
                except tf.errors.OutOfRangeError:
                    break
            psnr = np.mean(psnr_all)
            ssim = np.mean(ssim_all)
            print("Average psnr {}, Average ssim {}".format(psnr, ssim))
            out.close()


def main(argv):
    solver = TestSolver()
    solver.test(FLAGS.model_path)

if __name__ == "__main__":
    tf.app.run()
