import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--path',type=str,default='./Dataset', help='where you store your HSI data file')
parser.add_argument('--srfpath',type=str,default='NikonD700.npy', help='where you store your spectral response function ')
parser.add_argument('--hsi_size',type=tuple,default=(512,512,31), help='size of HR-HSI')
parser.add_argument('--save_path',type=str,default='QUI/', help='where you store your HSI data file')
parser.add_argument('--sf',type=int,default=32, help='scale_factor, set to 8, 16, and 32 in our experiment')
parser.add_argument('--pre_epoch',type=int,default=4000, help='scale_factor, set to 8, 16, and 32 in our experiment')
parser.add_argument('--max_epoch',type=int,default=4000, help='scale_factor, set to 8, 16, and 32 in our experiment')
parser.add_argument('--K', type=float, default=10, help='alpha')
args=parser.parse_args()
