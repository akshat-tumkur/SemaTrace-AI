"""Portable CLI for the one fixed practical adaptation."""
from pathlib import Path
import sys,argparse,json
if __package__ in (None,''):sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training.data import CODE

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['prepare','tokenize','status','smoke','pilot','train','calibrate','final','export'])
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--original-root',type=Path)
    parser.add_argument('--archives',type=Path);parser.add_argument('--selection',type=Path);parser.add_argument('--config',type=Path)
    args=parser.parse_args();root=args.root.resolve();root.mkdir(parents=True,exist_ok=True)
    from training_practical.engine import configuration
    config=configuration(root,args.config)
    if args.command=='prepare':
        from training_practical.data import prepare
        prepare(root,config,args.archives,args.original_root,args.selection)
    elif args.command=='tokenize':
        from training_practical.cache import encode
        encode(root,config)
    elif args.command=='status':
        from training_practical.data import frozen,verify_selection
        print(json.dumps({'partitions':frozen(root)['splits'],'subset':{k:v for k,v in verify_selection(root,config).items() if k!='entries'}},indent=2))
    elif args.command=='smoke':
        from training_practical.smoke import smoke
        smoke(root,config)
    elif args.command in ('pilot','train'):
        from training_practical.engine import pilot,train
        (pilot if args.command=='pilot' else train)(root,config)
    else:
        from training_practical import evaluation
        getattr(evaluation,args.command)(root,config)
if __name__=='__main__':main()
