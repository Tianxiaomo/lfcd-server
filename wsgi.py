import os
import time
import datetime
import cv2
import numpy as np
import uuid
import json
import math
import functools
import logging
import collections
from easydict import EasyDict as edict


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def get_config():
    config = edict()
    config.image_dir = 'data'
    config.checkpoints_dir = 'checkpoints'
    config.xml_dir = 'data'
    # hard nagative mining ratio, needed by loss layer
    config.param_hnm_ratio = 5

    # the number of image channels
    config.param_num_image_channel = 3

    # the number of output scales (loss branches)
    config.param_num_output_scales = 8

    # feature map size for each scale
    config.param_feature_map_size_list = [159, 159, 79, 79, 39, 19, 19, 19]

    # bbox lower bound for each scale
    config.param_bbox_small_list = [10, 15, 20, 40, 70, 110, 250, 400]
    assert len(config.param_bbox_small_list) == config.param_num_output_scales

    # bbox upper bound for each scale
    config.param_bbox_large_list = [15, 20, 40, 70, 110, 250, 400, 560]
    assert len(config.param_bbox_large_list) == config.param_num_output_scales

    # bbox gray lower bound for each scale
    config.param_bbox_small_gray_list = [math.floor(v * 0.9) for v in config.param_bbox_small_list]
    # bbox gray upper bound for each scale
    config.param_bbox_large_gray_list = [math.ceil(v * 1.1) for v in config.param_bbox_large_list]

    # the RF size of each scale used for normalization, here we use param_bbox_large_list for better regression
    config.param_receptive_field_list = config.param_bbox_large_list
    # RF stride for each scale
    config.param_receptive_field_stride = [4, 4, 8, 8, 16, 32, 32, 32]
    # the start location of the first RF of each scale
    config.param_receptive_field_center_start = [3, 3, 7, 7, 15, 31, 31, 31]

    config.param_normalization_constant = [i/(2.0*j) for i,j in zip(config.param_receptive_field_list,config.param_receptive_field_stride)]

    # the sum of the number of output channels, 2 channels for classification and 2 for bbox regression
    config.param_num_output_channels = 4

    config.num_workers = 1
    # config.batch_size = config.num_workers
    config.batch_size = 8
    config.gpus = '6'
    config.epochs = 4000
    config.lr = 1e-2
    config.optimizer = 'Adam'
    resume_epoch = 0
    # pre_weights = os.path.join(config.checkpoints_dir, 'ctpn_ep50_0.0075_0.0190_0.0264.pth.tar')
    config.pre_weights = None
    config.IMAGE_MEAN = [123.68, 116.779, 103.939]

    return config

class TextLineCfg:
    SCALE = 600
    MAX_SCALE = 1200
    TEXT_PROPOSALS_WIDTH = 16
    MIN_NUM_PROPOSALS = 2
    MIN_RATIO = 0.5
    LINE_MIN_SCORE = 0.9
    MAX_HORIZONTAL_GAP = 50
    TEXT_PROPOSALS_MIN_SCORE = 0.7
    TEXT_PROPOSALS_NMS_THRESH = 0.3
    MIN_V_OVERLAPS = 0.6
    MIN_SIZE_SIM = 0.6

class Graph:
    def __init__(self, graph):
        self.graph = graph

    def sub_graphs_connected(self):
        sub_graphs = []
        for index in range(self.graph.shape[0]):
            if not self.graph[:, index].any() and self.graph[index, :].any():
                v = index
                sub_graphs.append([v])
                while self.graph[v, :].any():
                    v = np.where(self.graph[v, :])[0][0]
                    sub_graphs[-1].append(v)
        return sub_graphs

class TextProposalGraphBuilder:
    """
        Build Text proposals into a graph.
    """

    def get_successions(self, index):
        box = self.text_proposals[index]
        results = []
        for left in range(int(box[0]) + 4, min(int(box[0]) + TextLineCfg.MAX_HORIZONTAL_GAP + 1, self.im_size[1]),4):
            adj_box_indices = self.boxes_table[left]
            for adj_box_index in adj_box_indices:
                if self.meet_v_iou(adj_box_index, index):
                    results.append(adj_box_index)
            if len(results) != 0:
                return results
        return results

    def get_precursors(self, index):
        box = self.text_proposals[index]
        results = []
        for left in range(int(box[0]) - 4, max(int(box[0] - TextLineCfg.MAX_HORIZONTAL_GAP), 0) - 1, -4):
            adj_box_indices = self.boxes_table[left]
            for adj_box_index in adj_box_indices:
                if self.meet_v_iou(adj_box_index, index):
                    results.append(adj_box_index)
            if len(results) != 0:
                return results
        return results

    def is_succession_node(self, index, succession_index):
        precursors = self.get_precursors(succession_index)
        if self.scores[index] >= np.max(self.scores[precursors]):
            return True
        return False

    def meet_v_iou(self, index1, index2):
        def overlaps_v(index1, index2):
            h1 = self.heights[index1]
            h2 = self.heights[index2]
            y0 = max(self.text_proposals[index2][1], self.text_proposals[index1][1])
            y1 = min(self.text_proposals[index2][3], self.text_proposals[index1][3])
            return max(0, y1 - y0 + 1) / min(h1, h2)

        def size_similarity(index1, index2):
            h1 = self.heights[index1]
            h2 = self.heights[index2]
            return min(h1, h2) / max(h1, h2)
        # print("index1:",index1,"    index2:",index2)
        # print(overlaps_v(index1, index2),"----",size_similarity(index1, index2))
        return overlaps_v(index1, index2) >= TextLineCfg.MIN_V_OVERLAPS and \
               size_similarity(index1, index2) >= TextLineCfg.MIN_SIZE_SIM

    def build_graph(self, text_proposals, scores, im_size):
        self.text_proposals = text_proposals
        self.scores = scores
        self.im_size = im_size
        self.heights = text_proposals[:, 3] - text_proposals[:, 1] + 1

        boxes_table = [[] for _ in range(self.im_size[1])]
        for index, box in enumerate(text_proposals):
            boxes_table[int(box[0])].append(index)
        self.boxes_table = boxes_table

        graph = np.zeros((text_proposals.shape[0], text_proposals.shape[0]), np.bool)

        for index, box in enumerate(text_proposals):
            successions = self.get_successions(index)
            if len(successions) == 0:
                continue
            succession_index = successions[np.argmax(scores[successions])]
            # print("succession_index{}__{}".format(index,succession_index));
            if self.is_succession_node(index, succession_index):
                # NOTE: a box can have multiple successions(precursors) if multiple successions(precursors)
                # have equal scores.
                # print("succession_index{}__{}".format(index, succession_index));
                graph[index, succession_index] = True
        return Graph(graph)

class TextProposalConnectorOriented:
    """
        Connect text proposals into text lines
    """

    def __init__(self):
        self.graph_builder = TextProposalGraphBuilder()

    def group_text_proposals(self, text_proposals, scores, im_size):
        # s = time.time()
        graph = self.graph_builder.build_graph(text_proposals, scores, im_size)
        # print(time.time()-s)
        return graph.sub_graphs_connected()

    def fit_y(self, X, Y, x1, x2):
        # len(X) != 0
        # if X only include one point, the function will get line y=Y[0]
        if np.sum(X == X[0]) == len(X):
            return Y[0], Y[0]
        p = np.poly1d(np.polyfit(X, Y, 1))
        return p(x1), p(x2)

    def get_text_lines(self, text_proposals, scores, im_size):
        """
        text_proposals:boxes

        """
        # tp=text proposal
        # s = time.time()
        tp_groups = self.group_text_proposals(text_proposals, scores, im_size)  # 首先还是建图，获取到文本行由哪几个小框构成
        # print("group_text_proposals",time.time()-s)
        # s = time.time()
        text_lines = np.zeros((len(tp_groups), 8), np.float32)

        for index, tp_indices in enumerate(tp_groups):
            text_line_boxes = text_proposals[list(tp_indices)]  # 每个文本行的全部小框
            X = (text_line_boxes[:, 0] + text_line_boxes[:, 2]) / 2  # 求每一个小框的中心x，y坐标
            Y = (text_line_boxes[:, 1] + text_line_boxes[:, 3]) / 2

            z1 = np.polyfit(X, Y, 1)  # 多项式拟合，根据之前求的中心店拟合一条直线（最小二乘）
            # print(z1)
            x0 = np.min(text_line_boxes[:, 0])  # 文本行x坐标最小值
            x1 = np.max(text_line_boxes[:, 2])  # 文本行x坐标最大值
            # print(x0,x1)
            offset = (text_line_boxes[0, 2] - text_line_boxes[0, 0]) * 0.5  # 小框宽度的一半

            # 以全部小框的左上角这个点去拟合一条直线，然后计算一下文本行x坐标的极左极右对应的y坐标
            lt_y, rt_y = self.fit_y(text_line_boxes[:, 0], text_line_boxes[:, 1], x0 + offset, x1 - offset)
            # 以全部小框的左下角这个点去拟合一条直线，然后计算一下文本行x坐标的极左极右对应的y坐标
            lb_y, rb_y = self.fit_y(text_line_boxes[:, 0], text_line_boxes[:, 3], x0 + offset, x1 - offset)

            score = scores[list(tp_indices)].sum() / float(len(tp_indices))  # 求全部小框得分的均值作为文本行的均值

            text_lines[index, 0] = x0
            text_lines[index, 1] = min(lt_y, rt_y)  # 文本行上端 线段 的y坐标的小值
            text_lines[index, 2] = x1
            text_lines[index, 3] = max(lb_y, rb_y)  # 文本行下端 线段 的y坐标的大值
            text_lines[index, 4] = score  # 文本行得分
            text_lines[index, 5] = z1[0]  # 根据中心点拟合的直线的k，b
            text_lines[index, 6] = z1[1]
            height = np.mean((text_line_boxes[:, 3] - text_line_boxes[:, 1]))  # 小框平均高度
            text_lines[index, 7] = height + 2.5

        text_recs = np.zeros((len(text_lines), 9), np.float)
        index = 0
        for line in text_lines:
            b1 = line[6] - line[7] / 2  # 根据高度和文本行中心线，求取文本行上下两条线的b值
            b2 = line[6] + line[7] / 2
            x1 = line[0]
            y1 = line[5] * line[0] + b1  # 左上
            x2 = line[2]
            y2 = line[5] * line[2] + b1  # 右上
            x3 = line[0]
            y3 = line[5] * line[0] + b2  # 左下
            x4 = line[2]
            y4 = line[5] * line[2] + b2  # 右下
            disX = x2 - x1
            disY = y2 - y1
            width = np.sqrt(disX * disX + disY * disY)  # 文本行宽度

            fTmp0 = y3 - y1  # 文本行高度
            fTmp1 = fTmp0 * disY / width
            x = np.fabs(fTmp1 * disX / width)  # 做补偿
            y = np.fabs(fTmp1 * disY / width)
            if line[5] < 0:
                x1 -= x
                y1 += y
                x4 += x
                y4 -= y
            else:
                x2 += x
                y2 += y
                x3 -= x
                y3 -= y
            text_recs[index, 0] = x1
            text_recs[index, 1] = y1
            text_recs[index, 2] = x2
            text_recs[index, 3] = y2
            text_recs[index, 4] = x3
            text_recs[index, 5] = y3
            text_recs[index, 6] = x4
            text_recs[index, 7] = y4
            text_recs[index, 8] = line[4]
            index = index + 1

        # print(time.time()-s)
        return text_recs

@functools.lru_cache(maxsize=1)
def get_host_info():
    ret = {}
    with open('/proc/cpuinfo') as f:
        ret['cpuinfo'] = f.read()

    with open('/proc/meminfo') as f:
        ret['meminfo'] = f.read()

    with open('/proc/loadavg') as f:
        ret['loadavg'] = f.read()

    return ret

@functools.lru_cache(maxsize=100)
def get_predictor(checkpoint_path):
    logger.info('loading model')
    from model import LFCDNet, LFCD_Unit
    import torch
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = LFCDNet()
    cc = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(cc['model_state_dict'], strict=True)
    model.to(device)
    model.eval()
    config = get_config()

    def predictor(img):
        """
        :return: {
            'text_lines': [
                {
                    'score': ,
                    'x0': ,
                    'y0': ,
                    'x1': ,
                    ...
                    'y3': ,
                }
            ],
            'rtparams': {  # runtime parameters
                'image_size': ,
                'working_size': ,
            },
            'timing': {
                'net': ,
                'restore': ,
                'nms': ,
                'cpuinfo': ,
                'meminfo': ,
                'uptime': ,
            }
        }
        """
        start_time = time.time()
        rtparams = collections.OrderedDict()
        rtparams['start_time'] = datetime.datetime.now().isoformat()
        rtparams['image_size'] = '{}x{}'.format(img.shape[1], img.shape[0])
        timer = collections.OrderedDict([
            ('net', 0),
            ('restore', 0),
            ('nms', 0)
        ])

        w,h = img.shape[0],img.shape[1]
        max_wh = max(w,h)
        if max_wh > 1600:
            max_wh = 1600
        if max(w,h) == w:
            new_w = (max_wh // 32)*32
            new_h = int(h*(new_w / w)//32 + 0.5)*32
        else:
            new_h = (max_wh // 32) * 32
            new_w = int(h*(new_h / h) // 32 + 0.5) * 32

        s_img = cv2.resize(img,(new_h,new_w))
        img = np.expand_dims(s_img, axis=0)
        img = (img - 127.5) / 127.5
        img = torch.from_numpy(img.transpose([0, 3, 1, 2])).float()
        img = img.to(device)

        rtparams['working_size'] = '{}x{}'.format(1,2)
        start = time.time()
        with torch.no_grad():
            out_cls, out_regr = model(img)
        timer['net'] = time.time() - start

        select_score_list = []
        select_anchor_list = []
        for i,(cls,regr) in enumerate(zip(out_cls,out_regr)):
            field_stride =  config.param_receptive_field_stride[i]
            normalization =  config.param_normalization_constant[i]
            out_bbox = regr[0].detach().numpy()
            a = np.where(cls[0, 0].detach().numpy() > -0.3)
            bbox = out_bbox[:, a[0], a[1]]
            y_min, y_max = (a[0]+0.5) * field_stride - bbox[0] * field_stride * normalization, (a[0]+0.5) * field_stride + bbox[1] * field_stride * normalization
            x_min, x_max = a[1] * field_stride, a[1] * field_stride + field_stride
            select_score = np.array([cls[0, 0].detach().numpy()[a]]).T
            select_anchor = np.array([x_min, y_min, x_max, y_max]).T
            select_score_list.append(select_score)
            select_anchor_list.append(select_anchor)

        select_anchor = np.concatenate(select_anchor_list).astype('int')
        select_score = np.concatenate(select_score_list)

        start = time.time()
        textConn = TextProposalConnectorOriented()
        text = textConn.get_text_lines(select_anchor, select_score, [new_w, new_h])
        timer['nms'] = time.time() - start

        logger.info('net {:.0f}ms, restore {:.0f}ms, nms {:.0f}ms'.format(
            timer['net']*1000, timer['restore']*1000, timer['nms']*1000))

        duration = time.time() - start_time
        timer['overall'] = duration
        logger.info('[timing] {}'.format(duration))

        text_lines = []
        if text is not None:
            for i in text:
                s = str(round(i[-1] * 100, 2)) + '%'
                i = [int(j) for j in i]
                cv2.line(s_img, (i[0], i[1]), (i[2], i[3]), (0, 0, 255), 2)
                cv2.line(s_img, (i[0], i[1]), (i[4], i[5]), (0, 0, 255), 2)
                cv2.line(s_img, (i[6], i[7]), (i[2], i[3]), (0, 0, 255), 2)
                cv2.line(s_img, (i[4], i[5]), (i[6], i[7]), (0, 0, 255), 2)
                text_lines.append([i[0],i[1],i[2],i[3],i[4],i[5],i[6],i[7],])

        ret = {
            'text_lines': text_lines,
            'rtparams': rtparams,
            'timing': timer,
        }
        ret.update(get_host_info())
        return ret,s_img

    return predictor

### the webserver
from flask import Flask, request, render_template
import argparse

class Config:
    SAVE_DIR = 'static/results'

config = Config()

application = Flask(__name__)

@application.route('/')
def index():
    return render_template('index.html', session_id='dummy_session_id')

def save_result(s_img,r_img, rst):
    session_id = str(uuid.uuid1())
    cur_path = os.path.abspath(os.path.dirname(__file__))
    dirpath = os.path.join(cur_path,config.SAVE_DIR, session_id)
    os.makedirs(dirpath)

    # save input image
    output_path = os.path.join(dirpath, 'input.png')
    cv2.imwrite(output_path, s_img)

    # save illustration
    output_path = os.path.join(dirpath, 'output.png')
    cv2.imwrite(output_path,r_img)

    # save json data
    output_path = os.path.join(dirpath, 'result.json')
    with open(output_path, 'w') as f:
        json.dump(rst, f)

    rst['session_id'] = session_id
    return rst

checkpoint_path = 'static/ctpn_ep660_2.0956_0.5052_2.6008.pth.tar'
cur_path = os.path.abspath(os.path.dirname(__file__))
checkpoint_path = os.path.join(cur_path,checkpoint_path)

@application.route('/', methods=['POST'])
def index_post():
    global predictor
    import io
    bio = io.BytesIO()
    request.files['image'].save(bio)
    img = cv2.imdecode(np.frombuffer(bio.getvalue(), dtype='uint8'), 1)
    rst,r_img = predictor(img)

    save_result(img,r_img,rst)
    return render_template('index.html', session_id=rst['session_id'])

predictor = get_predictor(checkpoint_path)

def main():
    global checkpoint_path
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', default=8769, type=int)
    parser.add_argument('--checkpoint_path', default=checkpoint_path)
    args = parser.parse_args()
    checkpoint_path = args.checkpoint_path

    if not os.path.exists(args.checkpoint_path):
        raise RuntimeError(
            'Checkpoint `{}` not found'.format(args.checkpoint_path))

    application.debug = False  # change this to True if you want to debug
    application.run('0.0.0.0', args.port)

if __name__ == '__main__':
    main()