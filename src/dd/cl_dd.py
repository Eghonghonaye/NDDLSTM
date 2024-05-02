import torch
import sys
from src.rl.Models.actorcritic import *
from torch import optim
from src.dd.solve import Solver
from src.utils.utils import *

def load_model(model_path,jobs,ops,device):
    input_model = './output_models/' + model_path + '.tar'
    embeddim = 128
    jobs = jobs
    ops = ops
    macs = ops

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
    return model


def run(args,model_path='base_50_20'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_csv = './output_logs/csv/test/ndd' + 'results' + '.csv'

    data_mode = 'ta'

    if data_mode == 'ta':
        max_data_size = 80

    for i in range(max_data_size):

        if data_mode == 'ta':
            problem_ind = i//10 + 2
            path = './data/ta/ta{}'.format(i+1)

            data = read_instances('./data/ta/ta{}'.format(i+1),full=True)
            jobs = data.njobs
            ops = data.nops

        if args.algorithm == "dd":
            DDSolver = Solver(path)
            solution = DDSolver.solveWithRerun(args)
            print(solution)
        elif args.algorithm == "dd-rl":
            model = load_model(model_path,jobs,ops,device)
            DDSolver = Solver(path)
            solution = DDSolver.solveWithRerun(args,True,model)
        else:
            print(f'You have supplied an unknown algorithm {args.algorithm}')
            sys.exit()

        with open(output_csv, 'a') as f:
            # print('%d, %.4f, %d'%(te_ite,test_reward,makespan), file=f)
            print(f'ta{i+1}, {jobs}, {ops}, {args.algorithm}, - ,{solution.bestSolution.state["max_span"]}',file=f)

    