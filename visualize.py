from matplotlib import pyplot as plt
plt.rcParams['toolbar'] = 'toolmanager'
from mpl_toolkits.axes_grid.axislines import SubplotZero
from matplotlib import image as im
from mpl_toolkits.mplot3d import Axes3D
from tkinter.filedialog import askdirectory
from matplotlib.backend_tools import ToolBase, ToolToggleBase
from math import pow, sqrt
import matplotlib.gridspec as gridspec

import numpy as np
import argparse
import os, cv2

FRAME_SIZE = 5
YMAX = XMAX = 8
TMAX = 5
x0 = [-1*(FRAME_SIZE//2)] * 9 + [0] * 9 + [FRAME_SIZE//2] * 9
y0 = [[-1*XMAX] * 3 + [0] * 3 + [XMAX] * 3] * 3
z0 = [-1*YMAX, 0 , YMAX] * 9  
DATA = {}
FIGURE = {}

class dirTools(ToolBase):
    default_keymap = "n"
    description = 'Open a new folder'

    def trigger(self, *args, **kwargs):
        d = askdirectory()
        all_files = os.listdir(d)
        input_frames = [x for x in all_files if x.endswith("in.png")]
        gt_frames = [x for x in all_files if x.endswith("in.png")]
        input_frames.sort()
        gt_frames.sort()        
        print("Selecting directory: ", d)
        print(len(input_frames))
        print(len(gt_frames))
        if not os.path.exists(os.path.join(d, "flow.npy")) or not os.path.exists(os.path.join(d, "flow_weights.npy")) \
            or len(input_frames) != FRAME_SIZE or len(gt_frames) != FRAME_SIZE:
            return False
        
        DATA["root_dir"] = d
        DATA["input"] = [cv2.imread(os.path.join(d, x)) for x in input_frames]
        DATA["gt"] = cv2.imread(os.path.join(d, "gt.png"))
        DATA["flow"] = np.transpose(np.load(os.path.join(d, "flow.npy")), (2, 0, 1, 3))
        DATA["weights"] = np.transpose(np.load(os.path.join(d, "flow_weights.npy")), (2, 0, 1))
        DATA["output"] = cv2.imread(os.path.join(d, "out.png"))
        
        flow_shape = np.shape(DATA["flow"])
        weight_shape = np.shape(DATA["weights"])
        assert flow_shape[0] == weight_shape[0]
        assert flow_shape[1] == weight_shape[1]
        assert flow_shape[2] == weight_shape[2]
        DATA["NUM_SAMPLE"] = flow_shape[0]
        DATA["H"] = flow_shape[1]
        DATA["W"] = flow_shape[2]

        image = FIGURE["image"]
        imgplot = image.imshow(DATA["input"][FRAME_SIZE//2], cmap='gray')
        DATA["cur_frame"] = FRAME_SIZE // 2
        plt.draw()

        global flow_printer 
        flow_printer = flowPrinter(imgplot)
        return True

class preFrame(ToolBase):
    default_keymap = ","
    description = "previous frame"

    def trigger(self, *args, **kwargs):
        cur_frame = DATA["cur_frame"]
        if cur_frame == 0:
            return
        implot = flow_printer.img
        implot.set_data(DATA["input"][cur_frame-1])
        DATA["cur_frame"] = cur_frame - 1
        plt.draw()

class nextFrame(ToolBase):
    default_keymap = "."
    description = "next frame"

    def trigger(self, *args, **kwargs):
        cur_frame = DATA["cur_frame"]
        if cur_frame == FRAME_SIZE-1:
            return
        implot = flow_printer.img
        implot.set_data(DATA["input"][cur_frame+1])
        DATA["cur_frame"] = cur_frame + 1
        plt.draw()
class SwitchTool1(ToolToggleBase):
    '''Hide lines with a given gid'''
    default_keymap = 'o'
    description = 'Change'

    def enable(self, *args):
        imgplot  = flow_printer.img
        imgplot.set_data(DATA["output"])
        plt.draw()

    def disable(self, *args):
        cur_frame = DATA["cur_frame"]
        implot = flow_printer.img
        implot.set_data(DATA["input"][cur_frame])
        plt.draw()

class SwitchTool2(ToolToggleBase):
    '''Hide lines with a given gid'''
    default_keymap = 'g'
    description = 'Change'

    def enable(self, *args):
        imgplot  = flow_printer.img
        imgplot.set_data(DATA["gt"])
        plt.draw()

    def disable(self, *args):
        implot = flow_printer.img
        implot.set_data(DATA["output"])
        plt.draw()


def plot_figure():
    gs = gridspec.GridSpec(3, 5)
    gs.update(left=0.05, right=0.95, wspace=0.05)
    image = plt.subplot(gs[:2, :3])

    flow_3d = plt.subplot(gs[:2, 3:], projection='3d')
    f3d = flow_3d.scatter([],[])
    flow_3d.set_xlabel('T frame')
    flow_3d.set_ylabel('W width')
    flow_3d.set_zlabel('H height')
    
    flow_2d_list = []
    for i in range(FRAME_SIZE):
        sub = plt.subplot(gs[2, i])
        flow_2d_list.append(sub)

    FIGURE["image"] = image
    FIGURE["flow_3d"] = flow_3d 
    FIGURE["flow_2d_list"] = flow_2d_list
    FIGURE["weight_text"] = plt.figtext(0.99, 0.01, 'footnote text', horizontalalignment='right')
    FIGURE["value_text"] = plt.figtext(0.01, 0.01, 'footnote text', horizontalalignment='left')

    return 

class flowPrinter:
    def __init__(self, img):
        self.img = img
        img.figure.canvas.mpl_connect('button_press_event', self.print_flow)
        img.figure.canvas.mpl_connect('motion_notify_event', self.print_weight)
        self.weight_text = FIGURE["weight_text"]
        self.value_text = FIGURE["value_text"]
        self.points = None
        self.weight = None

    def print_flow(self, event):
        # if in main image
        if event.inaxes==self.img.axes:
            x = int(event.xdata)
            y = int(event.ydata)
            # update 3d flow
            flow = DATA["flow"]
            print("FLOW at (%d, %d): (%f, %f, %f)" % (x, y, flow[0,y,x,0], flow[0,y,x,1], flow[0,y,x,2]))
            
            flow_3d = FIGURE["flow_3d"]
            flow_3d.clear()

            # flow_3d.scatter(list(flow[:, y, x, 2]), list(flow[:, y, x, 0]), list(flow[:, y, x, 1]))
            # flow_3d.scatter(x0, y0, z0, c="orange", marker='o')

            num_base = flow.shape[0] // 9
            color = ["red", "orange", "green", "blue", "purple"]
            for i in range(num_base):
                flow_3d.scatter(list(flow[i*9:(i+1)*9, y, x, 2]), list(flow[i*9:(i+1)*9, y, x, 0]), list(flow[i*9:(i+1)*9, y, x, 1]), c=color[i])
            
            flow_3d.scatter(x0, y0, z0, c="yellow", marker='o')
            
            flow_3d.set_xlabel('T frame')
            flow_3d.set_ylabel('W width')
            flow_3d.set_zlabel('H height')
            self.value_text.set_text("Input Value: %f Output Value: %f" % (DATA["input"][2][int(y), int(x)][0], DATA["output"][int(y), int(x)][0]))
            self.draw_2d_flow(x, y)
            self.save_2d_figure(x, y)
            plt.draw()

    def print_weight(self, event):
        frame_id = -1
        for i, f in enumerate(FIGURE["flow_2d_list"]): 
            if event.inaxes == f.axes:
                frame_id = i
                break
        if frame_id == -1:  return
        if self.points is None: return

        def euclid_dis(pos1, pos2):
            dis = pow(pos1[0]-pos2[0], 2) + pow(pos1[1]-pos2[1], 2) + pow(pos1[2]-pos2[2], 2)
            return sqrt(dis)
        x = event.xdata
        y = event.ydata
        pos = (x-8, y-8, frame_id-2)
        flow = self.points
        weights = self.weights
        min_dis = euclid_dis(pos, flow[0,:])
        min_id = 0
        for i in range (1, flow.shape[0]):
            if min_dis > euclid_dis(pos, flow[i,:]):
                min_dis = euclid_dis(pos, flow[i,:])
                min_id = i
        self.weight_text.set_text("value: %f" % self.part_img_value[frame_id][int(y),int(x)][0] + " weight: " + str(weights[min_id]))
        plt.draw()
        # print("Mouse event: mouse: ", pos, " closes: ", flow[min_id,:])
        # print("Weights: %f" % weigts[min_id])


    def draw_2d_flow(self, x, y):
        flow_2d = FIGURE["flow_2d_list"]
        gt_frames = DATA["input"]
        W_MAX = DATA["W"]
        H_MAX = DATA["H"]
        flow = DATA["flow"]
        weights = DATA["weights"]

        all_points = flow[:, y, x, :]
        all_weights = weights[:, y, x]
        self.points = all_points
        self.weights = all_weights
        self.part_img_value = []
        for i in range(FRAME_SIZE):
            t = np.float32(i - FRAME_SIZE // 2)
            # points = all_points[np.where(np.abs(all_points[:, 2] - t) < 0.5)]
            pos = []
            for j in range(DATA["NUM_SAMPLE"]):
                if abs(all_points[j, 2] - (i-FRAME_SIZE//2)) < 0.5:
                    pos.append(j)
            points = all_points[pos,:]
            w = all_weights[pos]
            print("Frame : %d :" % i, w)

            x_bottom = max(0, x-8)
            x_ceil = min(W_MAX, x+8)
            y_bottom = max(0, y-8)
            y_ceil = min(H_MAX, y+8)
            part_img = gt_frames[i][y_bottom:y_ceil, x_bottom:x_ceil]
            # print ("%f %f %f %f" % (x_bottom, x_ceil, y_bottom, y_ceil))
            flow_2d[i].clear()
            flow_2d[i].imshow(part_img, cmap='gray')
            flow_2d[i].scatter(list(points[:,0] + x - x_bottom), list(points[:,1] + y - y_bottom), c='r', s=6)
            flow_2d[i].scatter([x - x_bottom], [y - y_bottom], c='b', s=15, marker='x')

            self.part_img_value.append(part_img)

    def save_2d_figure(self, x, y):
        kpn_sample_points_t = [-2]*5+[-1]*5+[0]*5+[1]*5+[2]*5
        kpn_sample_points_y = [-2, -1, 0, 1, 2] * 5
        flow = DATA["flow"]
        plt.figure()
        plt.scatter(list(flow[:, y, x, 2]), list(flow[:, y, x, 0]), c="red", s=20)
        plt.scatter(kpn_sample_points_t, kpn_sample_points_y, color="blue", s=20, marker='x')
        plt.xticks(range(-2,3))
        plt.xlabel('Frames (0 as refernce frame)')
        plt.ylabel('Width axis (0 as sample point)')
        # plt.legend()
        plt.savefig("t_x.png", dpi=100)
        plt.close()
        plt.figure()
        plt.scatter(list(flow[:, y, x, 2]), list(flow[:, y, x, 1]), c="red", s=20)
        plt.scatter(kpn_sample_points_t, kpn_sample_points_y, color="blue", s=20, marker='x')
        plt.xticks(range(-2,3))
        plt.xlabel('Frames (0 as refernce frame)')
        plt.ylabel('Height axis (0 as sample point)')
        # plt.legend()
        plt.savefig("t_y.png", dpi=100)
        plt.close()

def print_weight(frame_cnt, event):
    print("mouse event on (%f, %f) with frame %d", event.xdata, event.ydata, frame_cnt)
    def euclid_dis(pos1, pos2):
        dis = pow(pos1[0]-pos1[0], 2) + pow(pos1[1]-pos1[1], 2) + pow(pos1[2]-pos1[2], 2)
        return sqrt(dis)
    if flow_printer.weights is None:
        return

    x = event.xdata
    y = event.ydata
    pos = (x-5, y-5, frame_cnt-2)

    flow = flow_printer.points
    weigts = flow_printer.weights
    min_dis = euclid_dis(pos, flow[0,:])
    min_id = 0
    
    for i in range (1, flow.shape[0]):
        if min_dis > euclid_dis(pos, flow[i,:]):
            min_dis = euclid_dis(pos, flow[i,:])
            min_id = i
    print("Mouse event: mouse: (%f, %f), closes: (%f, %f）" % (x, y, flow[min_id,0]+5, flow[min_id,1]+5))
    print("Weights: %f" % weigts[min_id])
            



fig = plt.figure()
plot_figure()

fig.canvas.manager.toolmanager.add_tool('Open', dirTools)
fig.canvas.manager.toolmanager.add_tool('Pre', preFrame)
fig.canvas.manager.toolmanager.add_tool('Next', nextFrame)
fig.canvas.manager.toolmanager.add_tool('Output', SwitchTool1)
fig.canvas.manager.toolmanager.add_tool('gt', SwitchTool2)
fig.canvas.manager.toolbar.add_tool('Open', 'navigation', 0)
fig.canvas.manager.toolbar.add_tool('Pre', 'navigation', 1)
fig.canvas.manager.toolbar.add_tool('Next', 'navigation', 2)
fig.canvas.manager.toolbar.add_tool('Output', 'navigation', 3)
fig.canvas.manager.toolbar.add_tool('gt', 'navigation', 4)
plt.show()
