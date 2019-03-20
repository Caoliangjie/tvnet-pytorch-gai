import os
import numpy as np
from data.frame_dataset import frame_dataset
from train_options import arguments
import torch.utils.data as data
from model.network import model
import scipy.io as sio
import cv2
from utils import *
import matplotlib.pyplot as plt
def save_flow_to_img_gai(flow, h, w, c):
    hsv = np.zeros((h, w, c), dtype=np.uint8)
    hsv[:, :, 0] = 225
    hsv[:, :, 2] = 225
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 1] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    #res_img_path = os.path.join('result', name)
    #cv2.imwrite(res_img_path, rgb)
    return rgb
i = 0
if __name__ == '__main__':
    assert torch.cuda.is_available(), "Only support GPU"
    args = arguments().parse()
    dataset = frame_dataset(args)
    args.data_size = [args.batch_size, 3, *(dataset.img_size)]
    dataloader = data.DataLoader(dataset)
    model = model(args).cuda()
    if not args.demo:
        for n_epoch in range(args.n_epoch):
            for i, data in enumerate(dataloader):
                model(data[0], data[1])
                model.optimize()
            if n_epoch % 100 == 99:
                print(n_epoch + 1)
            elif n_epoch < 100 and (n_epoch + 1) % 10 == 0:
                print(n_epoch + 1)
            elif n_epoch < 10:
                print(n_epoch + 1)
    for data in dataloader:
        u1, u2, x2_warp = model.forward(data[0], data[1], need_result=True)
        _, c, h, w = args.data_size
        u1_np = np.squeeze(u1.detach().cpu().data.numpy())
        u2_np = np.squeeze(u2.detach().cpu().data.numpy())
        flow_mat = np.zeros([h, w, 2])
        flow_mat[:, :, 0] = u1_np
        flow_mat[:, :, 1] = u2_np
        if not os.path.exists('result'):
           os.mkdir('result')
        res_mat_path = os.path.join('result', 'result.mat')
        sio.savemat(res_mat_path, {'flow': flow_mat})
        if args.visualize:
          rgb = save_flow_to_img_gai(flow_mat, h, w, c)
          cv2.imwrite('result/{}_result.png'.format(i), rgb)
          i = i + 1
    save_im_tensor(x2_warp.data, 'result/x2_warp.png')
    save_im_tensor(data[0], 'result/x1.png')
    save_im_tensor(data[1], 'result/x2.png')
