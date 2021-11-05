import torch
import argparse
import json
import os

LANGS='am,ar,bxr,cdo,cs,de,el,en,es,et,eu,fa,fi,fr,gn,hi,ht,hu,hy,id,ilo,is,ja,jv,ka,ko,kv,la,lv,mhr,mi,my,myv,pt,qu,ru,se,sw,tk,tr,vi,wo,xmf,zh,zh_yue'.split(',')
INFILE = 'outputs/udpos/weights-bert-base-multilingual-cased-MaxLen128_udpos_{}/{}_{}_s1_importances.pt'

def my_format(val):
    return str("{:.6f}".format(val))
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outfile', default='weights.txt')
    parser.add_argument('--split', default='dev')
    parser.add_argument('--target', default='mr')
    args = parser.parse_args()

    with open(args.outfile, 'w') as f:
        for lang in LANGS:
            infile = INFILE.format(lang, args.split, args.target)
            if not os.path.exists(infile):
                continue    
            importances = torch.load(infile)
            print(importances.size())
            exit()
            importances = torch.mean(importances, dim=-1)
            importances = importances.tolist()
            importances = [my_format(val) for val in importances]
            f.write(f'=============Language = {lang}=========================\n')
            f.write(json.dumps(importances)+'\n')
if __name__ == '__main__':
    main()