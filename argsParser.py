import re,argparse,sys
from src.utils.limits import(
    minint,
    maxint,
    infinity,
)

def createParser():
    parser = argparse.ArgumentParser(description='Analyse Results')
    parser.add_argument('--input', '-i',
                        type=str,
                        action='store',
                        default="data/ta/ta1",
                        help='XML Input File')
    parser.add_argument('--output', '-o',
                        type=str,
                        action='store',
                        default='result.txt',
                        required=False,
                        help='txt Schedule File')
    parser.add_argument('--algorithm', '-a',
                        type=str,
                        action='store',
                        default='dd',
                        required=False,
                        help='algorithm choice from [dd, dd-rl, dd-rl-graph, rl-train]')
    parser.add_argument('--timeout', '-t',
                        type=int,
                        action='store',
                        default=maxint,
                        required=False,
                        help='timeout in seconds')
    parser.add_argument('--iterations', '-it',
                        type=int,
                        action='store',
                        default=maxint,
                        required=False,
                        help='maximum iterations to explore')
    parser.add_argument('--model', '-m',
                        type=str,
                        action='store',
                        default=None,
                        required=False,
                        help='path to an already trained model to be used/saved by solver')
    parser.add_argument('--data_mode', '-dm',
                        type=str,
                        action='store',
                        default="ta",
                        required=False,
                        help='data mode choice from [ta, dmu]')
    return parser