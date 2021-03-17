import argparse
import os
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import _pickle as cPickle
import numpy as np

import config

MTL_PATH = "/home/zhiling/py3_workspace/freesound-audio-tagging-2019/dcase2019_task2/work/statistics/main_mtl/logmel_64frames_64melbins/train_source=curated_and_noisy/segment=5.0s,hop=2.0s,pad_type=repeat/holdout_fold=1"

ORG_PATH = "/home/zhiling/py3_workspace/freesound-audio-tagging-2019/dcase2019_task2/work/statistics/main/logmel_64frames_64melbins/train_source=curated_and_noisy/segment=5.0s,hop=2.0s,pad_type=repeat/holdout_fold=1"

def plot_results(args):

    # Arugments & parameters
    workspace = args.workspace
    train_source = args.train_source
    segment_seconds = args.segment_seconds
    hop_seconds = args.hop_seconds
    pad_type = args.pad_type
    mini_data = args.mini_data
    
    filename = 'main_mtl' if args.mtl else 'main'
    frames_per_second = config.frames_per_second
    mel_bins = config.mel_bins
    holdout_fold = 1
    max_plot_iteration = 50000
    iterations = np.arange(0, max_plot_iteration, 500)
    
    # Paths
    if mini_data:
        prefix = 'minidata_'
    else:
        prefix = ''
    
    if args.mtl:
        save_fig_path = 'train_source={}_results_mtl.png'.format(train_source)
    else:
        save_fig_path = 'train_source={}_results.png'.format(train_source)
    
    def _load_stat(model_type, target_source, idx=None):
        validate_statistics_path = os.path.join(workspace, 'statistics', filename, 
            '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
            'train_source={}'.format(train_source), 'segment={}s,hop={}s,pad_type={}'.format(
            segment_seconds, hop_seconds, pad_type), 'holdout_fold={}'.format(holdout_fold), 
            model_type, 'validate_statistics.pickle')
        
        validate_statistics_dict = cPickle.load(open(validate_statistics_path, 'rb'))
        
        average_precision = np.array([stat['average_precision'] for 
            stat in validate_statistics_dict[target_source]])

        per_class_lwlrap = np.array([stat['per_class_lwlrap'] for 
            stat in validate_statistics_dict[target_source]])

        weight_per_class = np.array([stat['weight_per_class'] for 
            stat in validate_statistics_dict[target_source]])
            
        lwlrap = np.sum(per_class_lwlrap * weight_per_class, axis=-1)
        mAP = np.mean(average_precision, axis=-1)
        
        if model_type.startswith("mtl_"):
            model_type = model_type[4:]
        legend = '{}'.format(model_type)
        
        results = {
            'mAP': mAP, 
            'lwlrap': lwlrap, 
            'legend': legend}
            
        print('Model: {}, target_source: {}'.format(model_type, target_source))
        # print('    mAP: {:.3f}'.format(mAP[-1]))
        # print('    lwlrap: {:.3f}'.format(lwlrap[-1]))
        idx = lwlrap.argmax() if idx is None else idx
        print('    mAP: {:.3f}'.format(mAP[idx]))
        print('    lwlrap: {:.3f}'.format(lwlrap[idx]))
        
        return results, idx
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    
    model_best_curated_idx = {}
    for n, target_source in enumerate(['curated', 'noisy']):
        lines = []
        # for model_type in ['Cnn_5layers_AvgPooling', 'Cnn_9layers_AvgPooling', 'Cnn_9layers_MaxPooling', 'Cnn_13layers_AvgPooling']:
        path0 = MTL_PATH if args.mtl else ORG_PATH
        for model_type in os.listdir(path0):
            if "GCNEmb" in model_type:
                continue
            try:
                results, idx = _load_stat(model_type, target_source=target_source, idx=model_best_curated_idx.get(model_type, None))
                model_best_curated_idx[model_type] = idx
                if results['legend'] == "Cnn_9layers_AvgPooling":
                    label = 'backbone'
                elif results['legend'] == "Cnn_9layers_AvgPooling2":
                    label = 'backbone2'
                elif results['legend'] == "ResNet_V101_AvgPooling":
                    label = 'resnet101'
                elif results['legend'].startswith("ResNet_V101_AvgPooling_GCNEmb"):
                    label = 'resnet101_' + re.search(r"[^_]+_[^_]+_[^_]+_(.+)", results['legend']).group(1)
                else:
                    label = re.search(r"[^_]+_[^_]+_[^_]+_(.+)", results['legend']).group(1)
                    # label = results['legend'][len("Cnn_9layers_AvgPooling_"):].strip()
                line, = axs[n].plot(results['lwlrap'], label=label)
                # line, = axs[n].plot(results['lwlrap'], label=results['legend'])
                lines.append(line)
            except Exception as e:
                print(e)
                continue
        
    axs[0].set_title('Target source: {}'.format('curated'))
    axs[1].set_title('Target source: {}'.format('noisy'))
    
    for i in range(2):
        axs[i].legend(handles=lines, loc=4)
        axs[i].set_ylim(0, 1.0)
        axs[i].set_xlabel('Iterations')
        axs[i].set_ylabel('lwlrap')
        axs[i].grid(color='b', linestyle='solid', linewidth=0.2)
        axs[i].xaxis.set_ticks(np.arange(0, len(iterations) + 1, len(iterations) // 4))
        axs[i].xaxis.set_ticklabels(np.arange(0, max_plot_iteration + 1, max_plot_iteration // 4))
    
    plt.tight_layout()
    plt.savefig(save_fig_path)
    print('Figure saved to {}'.format(save_fig_path))
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--workspace', type=str, required=True, help='Directory of your workspace.')
    parser.add_argument('--train_source', type=str, choices=['curated', 'noisy', 'curated_and_noisy'], required=True)
    parser.add_argument('--segment_seconds', type=float, required=True, help='Segment duration for training.')
    parser.add_argument('--hop_seconds', type=float, required=True, help='Hop duration between segments.')
    parser.add_argument('--pad_type', type=str, choices=['constant', 'repeat'], required=True, help='Pad short audio recordings with constant silence or repetition.')
    parser.add_argument('--mini_data', action='store_true', default=False, help='Set True for debugging on a small part of data.')
    parser.add_argument('--mtl', action='store_true', default=False)

    args = parser.parse_args()
    
    plot_results(args)