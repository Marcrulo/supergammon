import os
import gym
import sys
import csv
from agents import TDAgent, HumanAgent, TDAgentGNU, RandomAgent, evaluate_agents
from gnubg.gnubg_backgammon import GnubgInterface, GnubgEnv, evaluate_vs_gnubg
from gym_backgammon.envs.backgammon import WHITE, BLACK
from model import TDGammon, TDGammonCNN
from web_gui.gui import GUI
#from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

#  tensorboard --logdir=runs/ --host localhost --port 8001


def write_file(path, **kwargs):
    with open('{}/parameters.txt'.format(path), 'w+') as file:
        print("Parameters:")
        for key, value in kwargs.items():
            file.write("{}={}\n".format(key, value))
            print("{}={}".format(key, value))
        print()


def path_exists(path):
    if os.path.exists(path):
        return True
    else:
        print("The path {} doesn't exists".format(path))
        sys.exit()



# ==================================== TRAINING PARAMETERS ===================================
def args_train(args):
    save_step = args.save_step
    save_path = None
    n_episodes = args.episodes
    init_weights = args.init_weights
    lr = args.lr
    hidden_units = args.hidden_units
    lamda = args.lamda
    name = args.name
    model_type = args.type
    seed = args.seed
    gamma = args.gamma                  # NEW FEATURE
    hidden_layers = args.hidden_layers  # NEW FEATURE
    activation = args.activation        # NEW FEATURE
    

    eligibility = False
    optimizer = None

    if model_type == 'nn':
        net = TDGammon(hidden_units=hidden_units, lr=lr, lamda=lamda, gamma=gamma, hl=hidden_layers, activation=activation, init_weights=init_weights, seed=seed) # NEW FEATURE
        eligibility = True
        env = gym.make('gym_backgammon:backgammon-v0')

    else:
        net = TDGammonCNN(lr=lr, seed=seed)
        optimizer = True
        env = gym.make('gym_backgammon:backgammon-pixel-v0')

    if args.model and path_exists(args.model):
        # assert os.path.exists(args.model), print("The path {} doesn't exists".format(args.model))
        net.load(checkpoint_path=args.model, optimizer=optimizer, eligibility_traces=eligibility)

    if args.save_path and path_exists(args.save_path):
        # assert os.path.exists(args.save_path), print("The path {} doesn't exists".format(args.save_path))
        save_path = args.save_path

        write_file(
            save_path, save_path=args.save_path, type=model_type, hidden_units=hidden_units, hidden_layers=hidden_layers, gamma=gamma, alpha=net.lr, lamda=net.lamda,
            n_episodes=n_episodes, modules=[module for module in net.modules()]
        )

    net.train_agent(env=env, n_episodes=n_episodes, save_path=save_path, save_step=save_step, eligibility=eligibility, name_experiment=name)


# ==================================== WEB GUI PARAMETERS ====================================
def args_gui(args):
    if path_exists(args.model):
        # assert os.path.exists(args.model), print("The path {} doesn't exists".format(args.model))

        if args.type == 'nn':
            net = TDGammon(hidden_units=args.hidden_units, lr=0.1, lamda=None, init_weights=False)
            env = gym.make('gym_backgammon:backgammon-v0')
        else:
            net = TDGammonCNN(lr=0.0001)
            env = gym.make('gym_backgammon:backgammon-pixel-v0')

        net.load(checkpoint_path=args.model, optimizer=None, eligibility_traces=False)

        agents = {BLACK: TDAgent(BLACK, net=net), WHITE: HumanAgent(WHITE)}
        gui = GUI(env=env, host=args.host, port=args.port, agents=agents)
        gui.run()


# =================================== EVALUATE PARAMETERS ====================================
def args_evaluate(args):
    model_agent0 = args.model_agent0
    model_agent1 = args.model_agent1
    model_type = args.type
    hidden_units_agent0 = args.hidden_units_agent0
    hidden_units_agent1 = args.hidden_units_agent1
    n_episodes = args.episodes



    if path_exists(model_agent0) and path_exists(model_agent1):
        # assert os.path.exists(model_agent0), print("The path {} doesn't exists".format(model_agent0))
        # assert os.path.exists(model_agent1), print("The path {} doesn't exists".format(model_agent1))

        if model_type == 'nn':
            net0 = TDGammon(hidden_units=hidden_units_agent0, lr=0.1, lamda=None, init_weights=False)
            net1 = TDGammon(hidden_units=hidden_units_agent1, lr=0.1, lamda=None, init_weights=False)
            env = gym.make('gym_backgammon:backgammon-v0')
        else:
            net0 = TDGammonCNN(lr=0.0001)
            net1 = TDGammonCNN(lr=0.0001)
            env = gym.make('gym_backgammon:backgammon-pixel-v0')

        net0.load(checkpoint_path=model_agent0, optimizer=None, eligibility_traces=False)
        net1.load(checkpoint_path=model_agent1, optimizer=None, eligibility_traces=False)

        agents = {WHITE: TDAgent(WHITE, net=net1), BLACK: TDAgent(BLACK, net=net0)}

        evaluate_agents(agents, env, n_episodes)


# ===================================== GNUBG PARAMETERS =====================================
def args_gnubg(args):
    model_agent0 = args.model_agent0
    model_type = args.type
    hidden_units_agent0 = args.hidden_units_agent0
    n_episodes = args.episodes
    host = args.host
    port = args.port
    difficulty = args.difficulty
    iterations = args.iterations

    experiment = "/saved_models/"+model_agent0
    folder = os.getcwd() + experiment 
    directory = os.fsencode(folder)


    max_it_sizes = 0
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        
        # Find by chosen iteration amount
        if iterations:    
            if filename.endswith("{}.tar".format(str(iterations))):
                final_file = filename
                break
        
        # Otherwise the biggest iteration amount
        else:
            if filename.endswith(".tar"):
                size = filename.split('_')[-1][:-4]
                if int(size) > max_it_sizes:
                    max_it_sizes = int(size)

        if filename.endswith("{}.tar".format(str(max_it_sizes))):
            final_file = filename

    if path_exists(folder + '/' + final_file):
        # assert os.path.exists(model_agent0), print("The path {} doesn't exists".format(model_agent0))
        if model_type == 'nn':
            net0 = TDGammon(hidden_units=hidden_units_agent0, lr=0.1, lamda=None, init_weights=False)
        else:
            net0 = TDGammonCNN(lr=0.0001)

        net0.load(checkpoint_path=folder + '/' + final_file, optimizer=None, eligibility_traces=False)

        gnubg_interface = GnubgInterface(host=host, port=port)
        gnubg_env = GnubgEnv(gnubg_interface, difficulty=difficulty, model_type=model_type)
        evaluate_vs_gnubg(agent=TDAgentGNU(WHITE, net=net0, gnubg_interface=gnubg_interface), env=gnubg_env, n_episodes=n_episodes, difficulty=difficulty, model=model_agent0)



def args_stats(args, parser):
    exp = args.exp
    iterations = args.iterations


    experiment = "/saved_models/"+exp
    folder = os.getcwd() + experiment 
    directory = os.fsencode(folder)

    max_it_sizes = 0
    evaluation_dict = {
        "beginner"      :   -1,
        "intermediate"  :   -1,
        "advanced"      :   -1,
        "world"         :   -1
    }
    final_file = ""
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        
        # Evaluation
        if filename.startswith("evaluation"):
            diff = filename.split('_')[1]
            winrate = filename.split('_')[-1][:-4]
            evaluation_dict[diff] = winrate 

        # Find by chosen iteration amount
        if iterations:    
            if filename.endswith("{}.csv".format(str(iterations))) and filename.startswith("stats"):
                final_file = filename
                break
        
        # Otherwise the biggest iteration amount
        else:
            if filename.endswith(".csv") and filename.startswith("stats"):
                size = filename[6:-4]
                if int(size) > max_it_sizes:
                    max_it_sizes = int(size)

        if filename.endswith("{}.csv".format(str(max_it_sizes))):
            final_file = filename

    avg_td = []
    win_rates = []
    with open(folder+'/'+final_file) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:

            print(row)
            avg_td.append(float(row[10]))
            win_rates.append(float(row[5]))

    print("\nWINRATES:")
    print("Beginner:", evaluation_dict['beginner'])
    print("Intermediate:", evaluation_dict['intermediate'])
    print("Advanced:", evaluation_dict['advanced'])
    print("World_class:", evaluation_dict['world'])

    f = plt.figure(figsize=(10,4))
    ax1 = f.add_subplot(121)
    ax2 = f.add_subplot(122)
    ax1.plot(avg_td)
    ax2.plot(win_rates)
    ax1.title.set_text("Average TD-error \nfor each episode")
    ax2.title.set_text("Winrate (\"loss\")")
    plt.show()

# ===================================== PLOT PARAMETERS ======================================
def args_plot(args, parser):
    '''
    This method is used to plot the number of time an agent wins when it plays against an opponent.
    Instead of evaluating the agent during training (it can require some time and slow down the training), I decided to plot the wins separately, loading the different
    model saved during training.
    For example, suppose I run the training for 100 games and save my model every 10 games.
    Later I will load these 10 models, and for each of them, I will compute how many times the agent would win against an opponent.
    :return: None
    '''

    src = args.save_path
    hidden_units = args.hidden_units
    n_episodes = args.episodes
    opponents = args.opponent.split(',')
    host = args.host
    port = args.port
    difficulties = args.difficulty.split(',')
    model_type = args.type

    if path_exists(src):
        # assert os.path.exists(src), print("The path {} doesn't exists".format(src))

        for d in difficulties:
            if d not in ['beginner', 'intermediate', 'advanced', 'world_class']:
                parser.error("--difficulty should be (one or more of) 'beginner','intermediate', 'advanced' ,'world_class'")

        dst = args.dst

        if 'gnubg' in opponents and (not host or not port):
            parser.error("--host and --port are required when 'gnubg' is specified in --opponent")

        for root, dirs, files in os.walk(src):
            global_step = 0
            files = sorted(files)

            writer = SummaryWriter(dst)

            for file in files:
                if ".tar" in file:
                    print("\nLoad {}".format(os.path.join(root, file)))

                    if model_type == 'nn':
                        net = TDGammon(hidden_units=hidden_units, lr=0.1, lamda=None, init_weights=False)
                        env = gym.make('gym_backgammon:backgammon-v0')
                    else:
                        net = TDGammonCNN(lr=0.0001)
                        env = gym.make('gym_backgammon:backgammon-pixel-v0')

                    net.load(checkpoint_path=os.path.join(root, file), optimizer=None, eligibility_traces=False)

                    if 'gnubg' in opponents:
                        tag_scalar_dict = {}

                        gnubg_interface = GnubgInterface(host=host, port=port)

                        for difficulty in difficulties:
                            gnubg_env = GnubgEnv(gnubg_interface, difficulty=difficulty, model_type=model_type)
                            wins = evaluate_vs_gnubg(agent=TDAgentGNU(WHITE, net=net, gnubg_interface=gnubg_interface), env=gnubg_env, n_episodes=n_episodes)
                            tag_scalar_dict[difficulty] = wins[WHITE]

                        writer.add_scalars('wins_vs_gnubg/', tag_scalar_dict, global_step)

                        with open(root + '/results.txt', 'a') as f:
                            print("{};".format(file) + str(tag_scalar_dict), file=f)

                    if 'random' in opponents:
                        tag_scalar_dict = {}
                        agents = {WHITE: TDAgent(WHITE, net=net), BLACK: RandomAgent(BLACK)}
                        wins = evaluate_agents(agents, env, n_episodes)

                        tag_scalar_dict['random'] = wins[WHITE]

                        writer.add_scalars('wins_vs_random/', tag_scalar_dict, global_step)

                    global_step += 1

                    writer.close()
