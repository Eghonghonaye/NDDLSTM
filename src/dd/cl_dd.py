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

    data_mode = args.data_mode

    print(f"Using data mode: {data_mode}")

    if data_mode == 'ta' or data_mode == 'dmu':
        max_data_size = 80

    for i in range(max_data_size):
        output_txt = './output_logs/txt/test/ndd' + 'results' + '.txt'

        if data_mode == 'ta':
            problem_ind = i//10 + 2
            path = './data/ta/ta{}'.format(i+1)

            data = read_instances('./data/ta/ta{}'.format(i+1),full=True)
            jobs = data.njobs
            ops = data.nops

        if data_mode == 'dmu':
            problem_ind = i//10 + 2
            path = f"./data/dmu/dmu{i+1}.txt"

            data = read_instances(f"./data/dmu/dmu{i+1}.txt",full=True)
            jobs = data.njobs
            ops = data.nops

        if args.algorithm == "dd":
            DDSolver = Solver(path)
            solution = DDSolver.solveWithRerun(args)
            
            with open(output_csv, 'a') as f:
                # print('%d, %.4f, %d'%(te_ite,test_reward,makespan), file=f)
                if data_mode == 'ta':
                    print(f'ta{i+1}, {jobs}, {ops}, {args.algorithm}, - ,{solution.bestSolution.state["max_span"]}',file=f)
                if data_mode == 'dmu':
                    print(f'dmu{i+1}, {jobs}, {ops}, {args.algorithm}, - ,{solution.bestSolution.state["max_span"]}',file=f)

            with open(output_txt, 'a') as f:
                if data_mode == 'ta':
                    print(f'ta{i+1}, {jobs}, {ops}, {args.algorithm}, - ,{solution.bestSolution.state["placement"].tolist()}',file=f)
                if data_mode == 'dmu':
                    print(f'dmu{i+1}, {jobs}, {ops}, {args.algorithm}, - ,{solution.bestSolution.state["placement"].tolist()}',file=f)
                # np.savetxt(output_txt, solution.bestSolution.state["placement"], fmt='%d')

        elif args.algorithm == "dd-rl":
            model = load_model(model_path,jobs,ops,device)
            DDSolver = Solver(path)
            solution = DDSolver.solveWithRerun(args,True,model)

            with open(output_csv, 'a') as f:
                # print('%d, %.4f, %d'%(te_ite,test_reward,makespan), file=f)
                if data_mode == 'ta':
                    print(f'ta{i+1}, {jobs}, {ops}, {args.algorithm}, - ,{solution.bestSolution.state["max_span"]}',file=f)
                if data_mode == 'dmu':
                    print(f'dmu{i+1}, {jobs}, {ops}, {args.algorithm}, - ,{solution.bestSolution.state["max_span"]}',file=f)

            with open(output_txt, 'a') as f:
                if data_mode == 'ta':
                    print(f'ta{i+1}, {jobs}, {ops}, {args.algorithm}, - ,{solution.bestSolution.state["placement"].tolist()}',file=f)
                if data_mode == 'dmu':
                    print(f'dmu{i+1}, {jobs}, {ops}, {args.algorithm}, - ,{solution.bestSolution.state["placement"].tolist()}',file=f)
                # np.savetxt(output_txt, solution.bestSolution.state["placement"], fmt='%d')

        else:
            print(f'You have supplied an unknown algorithm {args.algorithm}')
            sys.exit()

        


    