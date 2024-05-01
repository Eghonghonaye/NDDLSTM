from argsParser import createParser
import sys
from src.dd.solve import Solver

def main():
    # parse args
    parser = createParser()
    args = parser.parse_args()

    if args.algorithm == "dd":
        DDSolver = Solver(args.input)
        solution = DDSolver.solveWithRerun(args)
        print(solution)
    elif args.algorithm == "dd-rl":
        pass
    elif args.algorithm == "rl":
        pass
    else:
        print(f'You have supplied an unknown algorithm {args.algorithm}')
        sys.exit()



if __name__ == "__main__":
    main()