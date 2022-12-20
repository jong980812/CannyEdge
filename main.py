import argparse
import os
from pathlib import Path

def get_args():
    parser = argparse.ArgumentParser('VideoMAE fine-tuning and evaluation script for video classification', add_help=False)
    parser.add_argument('--run_dir', default='./', type=str,                               help='Dir python file' )
    parser.add_argument('--smoothing', default=None, choices=['gaussian', 'avg'], type=str, help='Choice Smoothing method')
    parser.add_argument('--sigma', default=1.0, type=float,                                 help='Set Standard deviation')
    parser.add_argument('--kernel_size', default=None, type=int,                               help='Direct setting Kernel size')
    parser.add_argument('--epsilon', default=0.01, type=float,                              help='Set Threshold Value to determine Kernel size')
    parser.add_argument('--gradient', default='sobel', 
                       choices=['roberts','sobel','prewitt','LoG','DoG'],                   help='Choose Mehotd to calculate gradient')    
    

    return parser.parse_args()

def main(args):
    print(args)



if __name__== "__main__":
    args=get_args()
    if not os.path.isdir(os.path.join(args.run_dir,'result')):  
        #@ Check output Diretory.
        '''
        If there is no 'result' directory in your run directory, 
        Make the result folder.
        '''
        output_path=os.path.join(args.run_dir,'result')
        os.mkdir(output_path)
        main(args)
    else:
        main(args)
        