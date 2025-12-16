import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
print("mymain")


import sys,argparse
print("mymain")
import NCSN_train.main_ImageNet as ncsn
#from NCSN_train import main_ImagNet
print("mymain")
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='ncsn', help='The model to use as a prior')
    parsed,sys.argv = parser.parse_known_args()
    sys.argv.insert(0,parsed.model)
    
    print(parsed)

    if parsed.model == 'ncsn':
        print("mymain")
        ncsn.main()
        
    elif parsed.model == 'glow':
        glow.main()
    else:
        print('Unknown model \'{}\': please select a model from the list: {}'.format(parsed.model, ['ncsn','glow']))
