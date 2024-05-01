from argsParser import createParser
import sys
from src.dd.solve import Solver
from main_test import main as testrl
from src.rl.Models.actorcritic import *
from torch import optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    # parse args
    parser = createParser()
    args = parser.parse_args()

    if args.algorithm == "dd":
        DDSolver = Solver(args.input)
        solution = DDSolver.solveWithRerun(args)
        print(solution)
    elif args.algorithm == "dd-rl":
        input_model = './output_models/' + 'base_50_20' + '.tar'
        embeddim = 128
        jobs = 15
        ops = 15
        macs = 15

        # Instanitate actor-critic
        actor = Actor(embeddim,jobs,ops,macs,device).to(device)
        critic = Critic(embeddim,jobs,ops,macs,device).to(device)

        # Environment training

        actor_opt = optim.Adam(actor.parameters())
        critic_opt = optim.Adam(critic.parameters())

        checkpoint = torch.load(input_model, map_location=torch.device('cpu'))
        actor.load_state_dict(checkpoint['actor_state_dict'])
        critic.load_state_dict(checkpoint['critic_state_dict'])
        actor_opt.load_state_dict(checkpoint['actor_opt_state_dict'])
        critic_opt.load_state_dict(checkpoint['critic_opt_state_dict'])

        model = {"actor":actor,"critic":critic}
        DDSolver = Solver(args.input)
        solution = DDSolver.solveWithRerun(args,True,model)
    elif args.algorithm == "rl":
        testrl()
    else:
        print(f'You have supplied an unknown algorithm {args.algorithm}')
        sys.exit()



if __name__ == "__main__":
    main()