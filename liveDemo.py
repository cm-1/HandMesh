import os.path as osp
import torch
import torch.backends.cudnn as cudnn
from cmr.cmr_sg import CMR_SG
from cmr.cmr_pg import CMR_PG
from cmr.cmr_g import CMR_G
from mobrecon.mobrecon_densestack import MobRecon
from utils.read import spiral_tramsform
from utils import utils, writer
from options.base_options import BaseOptions
from datasets.FreiHAND.freihand import FreiHAND
from datasets.Human36M.human36m import Human36M
from torch.utils.data import DataLoader
from run import Runner

import cv2


class DefaultArgsWrapper(dict):
    # Methods to allow me to call dict.key instead of dict["key"]
    # Inspired by https://stackoverflow.com/a/23689767
    # However, there are ways to improve it, as the comments suggest.
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def getRunner(
    focalLength,
    refWidthForFocalLength,
    exp_name="mobrecon",
    backbone="DenseStack",
    dataset="FreiHAND",
    model="mobrecon",
    device_idx = [-1],
    resume="mobrecon_densestack_dsconv.pt"
):
    parser, originalArgs = BaseOptions().getArgsAndParser()

    defaultArgsDict = {key: parser.get_default(key) for key in vars(originalArgs)}

    args = DefaultArgsWrapper(defaultArgsDict)

    args.exp_name = exp_name
    args.backbone = backbone
    args.dataset = dataset
    args.model = model
    args.device_idx = device_idx
    args.resume = resume
    if model == "mobrecon":
        args.size = 128
        args.out_channels = [32, 64, 128, 256]
        args.seq_length = [9, 9, 9, 9]

    args.work_dir = osp.dirname(osp.realpath(__file__))
    data_fp = osp.join(args.work_dir, 'data', args.dataset)
    args.out_dir = osp.join(args.work_dir, 'out', args.dataset, args.exp_name)
    args.checkpoints_dir = osp.join(args.out_dir, 'checkpoints')

    # device set
    if -1 in args.device_idx or not torch.cuda.is_available():
        device = torch.device('cpu')
    elif len(args.device_idx) == 1:
        device = torch.device('cuda', args.device_idx[0])
    else:
        device = torch.device('cuda')
    torch.set_num_threads(args.n_threads)

    # deterministic
    cudnn.benchmark = True
    cudnn.deterministic = True

    if args.dataset=='Human36M':
        template_fp = osp.join(args.work_dir, 'template', 'template_body.ply')
        transform_fp = osp.join(args.work_dir, 'template', 'transform_body.pkl')
    else:
        template_fp = osp.join(args.work_dir, 'template', 'template.ply')
        transform_fp = osp.join(args.work_dir, 'template', 'transform.pkl')
    spiral_indices_list, down_transform_list, up_transform_list, tmp = spiral_tramsform(transform_fp, template_fp, args.ds_factors, args.seq_length, args.dilation)

    # model
    if args.model == 'cmr_sg':
        model = CMR_SG(args, spiral_indices_list, up_transform_list)
    elif args.model == 'cmr_pg':
        model = CMR_PG(args, spiral_indices_list, up_transform_list)
    elif args.model == 'cmr_g':
        model = CMR_G(args, spiral_indices_list, up_transform_list)
    elif args.model == 'mobrecon':
        for i in range(len(up_transform_list)):
            up_transform_list[i] = (*up_transform_list[i]._indices(), up_transform_list[i]._values())
        model = MobRecon(args, spiral_indices_list, up_transform_list)
    else:
        raise Exception('Model {} not support'.format(args.model))

    # load
    epoch = 0
    if args.resume:
        if len(args.resume.split('/')) > 1:
            model_path = args.resume
        else:
            model_path = osp.join(args.checkpoints_dir, args.resume)
        checkpoint = torch.load(model_path, map_location='cpu')
        if checkpoint.get('model_state_dict', None) is not None:
            checkpoint = checkpoint['model_state_dict']
        model.load_state_dict(checkpoint)
        epoch = checkpoint.get('epoch', -1) + 1
    model = model.to(device)

    # run
    runner = Runner(args, model, tmp['face'], device)

    runner.set_demo(args)
    runner.model.eval()
    runner.setKMatrix(focalLength, refWidthForFocalLength)
    return runner

class webcamManager:
    def __init__(self, runner, width, height):
        self.runner = runner
        cap = cv2.VideoCapture(0) #,cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap = cap
        self.step = 0

    def displayFrame(self):
        ret, frame = self.cap.read()

        convertedFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if not ret:
            return

        handInfo = self.runner.processFrame(convertedFrame, self.step)
        colourList = [(0, 0, 255), (0, 255, 0)]
        for i in range(handInfo.points2D.shape[0]):
            
            for j in range(handInfo.points2D.shape[1]):
                pt = (handInfo.points2D[i][j]).astype(int)

                circleRadius = 5
                colour = colourList[i]
                thickness = 1
                
                frame = cv2.circle(frame, pt, circleRadius, colour, thickness)

        cv2.imshow("test", frame)
        self.step += 1



    

print("Starting")
w = 640
h = 480
r = getRunner(555, w)
wcm = webcamManager(r, w, h)
while True:
    wcm.displayFrame()
    # Break on press of "Q" key.
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
print("Done")

