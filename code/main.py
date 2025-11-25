import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import world
import utils
from world import cprint
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
import Procedure
from os.path import join
from time import strftime
# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset

if not os.path.exists('logs'):
    os.mkdir('logs')
    
if not os.path.exists('embs'):
    os.mkdir('embs')
    
log_template = f'{world.args.dataset}_seed{world.args.seed}_{world.args.model}_dim{world.args.recdim}_lr{world.args.lr}_dec{world.args.decay}'
run_datetime = strftime("%Y%m%d-%H%M%S")
file_prefix = f'{log_template}_{run_datetime}'
log_path = f'logs/{file_prefix}.txt'

Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)
bpr = utils.BPRLoss(Recmodel, world.config)

weight_file = utils.getFileName()
print(f"load and save to {weight_file}")
if world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
        world.cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")
Neg_k = 1

# init tensorboard
if world.tensorboard:
    w : SummaryWriter = SummaryWriter(
                                    join(world.BOARD_PATH, time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
                                    )
else:
    w = None
    world.cprint("not enable tensorflowboard")

try:
    for epoch in range(world.TRAIN_epochs):
        start = time.time()
        if epoch % 5 == 0:
            cprint("[TEST]")
            test_results = Procedure.Test(dataset, Recmodel, epoch, w, world.config['multicore'])
            
            if len(world.topks) == 2:
                test_log = [test_results['ndcg'][0], test_results['ndcg'][1],
                            test_results['recall'][0], test_results['recall'][1],
                            test_results['precision'][0], test_results['precision'][1]]
            elif len(world.topks) == 1:
                test_log = [test_results['ndcg'][0],
                            test_results['recall'][0],
                            test_results['precision'][0]]
            
            with open(log_path, 'a') as f:
                f.write(f'test ' + ' '.join([str(x) for x in test_log]) + '\n')
                
        output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k,w=w)
        print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}')
        torch.save(Recmodel.state_dict(), weight_file)
finally:
    if world.tensorboard:
        w.close()