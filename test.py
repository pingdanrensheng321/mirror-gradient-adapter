import argparse
import os

import yaml
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import datasets
import models
import utils
from torchstat import stat

from torchvision import transforms
from mmcv.runner import load_checkpoint

from thop import profile
import time
import numpy as np

def batched_predict(model, inp, coord, bsize):
    with torch.no_grad():
        model.gen_feat(inp)
        n = coord.shape[1]
        ql = 0
        preds = []
        while ql < n:
            qr = min(ql + bsize, n)
            pred = model.query_rgb(coord[:, ql: qr, :])
            preds.append(pred)
            ql = qr
        pred = torch.cat(preds, dim=1)
    return pred, preds


def tensor2PIL(tensor):
    toPIL = transforms.ToPILImage()
    return toPIL(tensor)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def eval_psnr(loader, model, data_norm=None, eval_type=None, eval_bsize=None,verbose=False):
    model.eval()
    if data_norm is None:
        data_norm = {
            'inp': {'sub': [0], 'div': [1]},
            'gt': {'sub': [0], 'div': [1]}
        }
    save_img_fig= './tmp/'+eval_type
    save_img_fig_gt= './tmp/'+eval_type+'/gt'
    if not os.path.exists(save_img_fig):       
        os.makedirs(save_img_fig)
    # import pdb;pdb.set_trace()
    # if eval_type == 'f1':
    #     metric_fn = utils.calc_f1
    #     metric1, metric2, metric3, metric4 = 'f1', 'auc', 'none', 'none'
    if eval_type == 'fmeasure':
        metric_fn = utils.calc_fmeasure
        metric1, metric2, metric3, metric4 = 'f_mea', 'mae', 'none', 'none'
        if not os.path.exists(save_img_fig):       
            os.makedirs(save_img_fig)
        if not os.path.exists(save_img_fig_gt):       
            os.makedirs(save_img_fig_gt)
    # elif eval_type == 'ber':
    #     metric_fn = utils.calc_ber
    #     metric1, metric2, metric3, metric4 = 'shadow', 'non_shadow', 'ber', 'none'
    # elif eval_type == 'cod':
    #     metric_fn = utils.calc_cod
    #     metric1, metric2, metric3, metric4 = 'sm', 'em', 'wfm', 'mae'
    elif eval_type == 'mirror':
        metric_fn = utils.calc_glass
        metric1, metric2, metric3, metric4 = 'iou', 'mae', 'ber', 'f_mea'
        save_img_fig= './tmp/'+eval_type
        if not os.path.exists(save_img_fig):       
            os.makedirs(save_img_fig)
        if not os.path.exists(save_img_fig_gt):       
            os.makedirs(save_img_fig_gt)
    elif eval_type == 'iou':
        metric_fn = utils.calc_iou
        metric1, metric2, metric3, metric4 = 'iou', 'iou', 'iou', 'iou'
        save_img_fig= './tmp/'+eval_type
        if not os.path.exists(save_img_fig):       
            os.makedirs(save_img_fig)
        if not os.path.exists(save_img_fig_gt):       
            os.makedirs(save_img_fig_gt)
    val_metric1 = utils.Averager()
    val_metric2 = utils.Averager()
    val_metric3 = utils.Averager()
    val_metric4 = utils.Averager()
    
    pbar = tqdm(loader, leave=False, desc='val')
    cnt= 0
    for batch in pbar:
        for k, v in batch.items():
            batch[k] = v.cuda()
        inp = batch['inp']
        # import pdb;pdb.set_trace()

        # t_all = []
        # for i in range(100):
        #     t1 = time.time()
        #     han = model.infer(inp)
        #     t2 = time.time()
        #     t_all.append(t2 - t1)

        # print('average time:', np.mean(t_all) / 1)
        # print('average fps:',1 / np.mean(t_all))

        # import pdb;pdb.set_trace()



        pred = torch.sigmoid(model.infer(inp))
        # import pdb;pdb.set_trace()
        result1, result2, result3, result4 = metric_fn(pred, batch['gt'])
        
        val_metric1.add(result1.item(), inp.shape[0])
        val_metric2.add(result2.item(), inp.shape[0])
        val_metric3.add(result3.item(), inp.shape[0])
        val_metric4.add(result4.item(), inp.shape[0])
        # print('iou',result1.item())
        # print('mae',result2.item())
        # print('ber',result3.item())
        # print('f_mea',result4.item())
        # import pdb;pdb.set_trace()
        if verbose:
            pbar.set_description('val {} {:.4f}| {} {:.4f}| {} {:.4f}| {} {:.4f}'.format(metric1, val_metric1.item(),metric2, val_metric2.item(),metric3, val_metric3.item(),metric4, val_metric4.item()))
            # pbar.set_description('val {} {:.4f}'.format(metric2, val_metric2.item()))
            # pbar.set_description('val {} {:.4f}'.format(metric3, val_metric3.item()))
            # pbar.set_description('val {} {:.4f}'.format(metric4, val_metric4.item()))
        with torch.no_grad():
            for p in range(pred.shape[0]):
                # import pdb;pdb.set_trace()
                img_save = pred[p]>0.5
                image_name = os.path.basename(loader.dataset.dataset.dataset_1.files[cnt])
                pil = tensor2PIL((255.0*img_save).to(torch.int)).convert('L')
                pil.save(save_img_fig + f'/{cnt}_pred_{image_name}')
                pil = tensor2PIL((255*batch['gt'][p]).to(torch.uint8))
                pil.save(save_img_fig_gt + f'/{cnt}_pred_{image_name}')
                cnt+=1
            # for p in range(pred.shape[0]):
            #     pil = tensor2PIL((255 * pred[p]).clone().detach().requires_grad_(True))
            #     pil.save(f'tmp/{cnt}_pred-{p}.png')
            #     pil = tensor2PIL((255 * batch['gt'][p]).clone().detach().requires_grad_(True).to(torch.uint8))
            #     pil.save(f'tmp/{cnt}_gt-{p}.png')
            #     cnt += 1
    # import pdb;pdb.set_trace()
    return val_metric1.item(), val_metric2.item(), val_metric3.item(), val_metric4.item()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--model')
    parser.add_argument('--prompt', default='none')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    spec = config['test_dataset']
    dataset = datasets.make(spec['dataset'])
    dataset = datasets.make(spec['wrapper'], args={'dataset': dataset})
    loader = DataLoader(dataset, batch_size=spec['batch_size'],
                        num_workers=8)

    model = models.make(config['model']).cuda()
    # import pdb;pdb.set_trace()
    sam_checkpoint = torch.load(args.model, map_location='cuda:0')
    model.load_state_dict(sam_checkpoint, strict=False)

    # model_keys = model.state_dict().keys()

    # # 过滤 checkpoint，只保留和模型结构匹配的键
    # filtered_checkpoint = {k: v for k, v in sam_checkpoint.items() if k in model_keys}

    # 加载过滤后的 checkpoint（严格模式就不会报错了）
    # model.load_state_dict(filtered_checkpoint, strict=True)
    
    
    
    with torch.no_grad():
        metric1, metric2, metric3, metric4 = eval_psnr(loader, model,
                                                    data_norm=config.get('data_norm'),
                                                    eval_type=config.get('eval_type'),
                                                    eval_bsize=config.get('eval_bsize'),
                                                    verbose=True)
    print('metric1: {:.4f}'.format(metric1))
    print('metric2: {:.4f}'.format(metric2))
    print('metric3: {:.4f}'.format(metric3))
    print('metric4: {:.4f}'.format(metric4))
