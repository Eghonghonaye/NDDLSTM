from argsParser import createParser
import sys
from main_test import main as testrl
from src.rl.Models.actorcritic import *
from torch import optim
from src.dd.cl_dd import run as runDD

def main():
    # parse args
    parser = createParser()
    args = parser.parse_args()

    if args.algorithm == "dd":
        runDD(args)
    elif args.algorithm == "dd-rl":
        runDD(args)
    elif args.algorithm == "rl":
        testrl()
    else:
        print(f'You have supplied an unknown algorithm {args.algorithm}')
        sys.exit()



if __name__ == "__main__":
    main()