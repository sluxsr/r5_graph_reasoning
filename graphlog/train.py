import os
import random
import numpy as np
from argparse import ArgumentParser
from collections import defaultdict, deque

from inits import get_rel_path
from train_dl import *
from dedution import Allpair, Deduction
from mcts import MCTSAgent
from policy_value_net import PolicyValueNet

# import certifi
# import urllib3
# http = urllib3.PoolManager(
#      cert_reqs='CERT_REQUIRED',
#      ca_certs=certifi.where()
# )

class Trainer():
    def __init__(self, hparams): 
        
        self.dataloader = TrainLoader(hparams)
        self.hparams = hparams
        self.rules = {}
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.allpair = Allpair(width=hparams.n_rel + hparams.n_unkn)
        self.deduction = Deduction(self.rules, self.allpair, hparams)

        self.learn_rate = hparams.learning_rate
        self.lr_multiplier = hparams.lr_multiplier
        self.temp = hparams.temp  
        self.n_playout = hparams.n_playout  
        self.c_puct = hparams.c_puct
        self.buffer_size = hparams.buffer_size
        self.batch_size = hparams.batch_size  
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.epochs = hparams.epochs
        self.kl_targ = hparams.kl_targ

        self.policy_value_net = PolicyValueNet(hparams.n_rel + hparams.n_unkn,
                                               hparams.n_rel + hparams.n_unkn)
        self.mcts_agent = MCTSAgent(self.policy_value_net.policy_value_fn,
                                      c_puct=self.c_puct,
                                      n_playout=self.n_playout)

    def get_traindl(self):

        train_set = self.dataloader.get_train_dataset()
        traindl =  DataLoader(
                dataset=train_set,
                collate_fn=pyg_collate,
                num_workers=8,
                batch_size=1,
                shuffle=True,
            )
        return traindl
    
    def get_testdl(self):

        test_set = self.dataloader.get_test_dataset()
        testdl =  DataLoader(
                dataset=test_set,
                collate_fn=pyg_collate,
                num_workers=8,
                batch_size=1,
                shuffle=True,
            )
        return testdl

    def collect_deduction_data(self, batch):
        _, deduction_data = self.deduction.start_deduction(  batch,
                                                        self.mcts_agent,
                                                        temp=self.temp)
        deduction_data = list(deduction_data)[:]
        self.episode_len = len(deduction_data)
        # augment the data
        deduction_data = self.get_equi_data(deduction_data)
        self.data_buffer.extend(deduction_data)

    def get_equi_data(self, deduction_data):
        extend_data = []
        for state, mcts_porb, z in deduction_data:
            for i in [1, 2, 3, 4]:
                # rotate counterclockwise
                equi_state = np.array([np.rot90(s, i) for s in state])
                equi_mcts_prob = np.rot90(np.flipud(
                    mcts_porb.reshape(self.hparams.n_rel + self.hparams.n_unkn, self.hparams.n_rel + self.hparams.n_unkn)), i)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    z))
                # flip horizontally
                equi_state = np.array([np.fliplr(s) for s in equi_state])
                equi_mcts_prob = np.fliplr(equi_mcts_prob)
                extend_data.append((equi_state,
                                    np.flipud(equi_mcts_prob).flatten(),
                                    z))
        return extend_data

    def policy_update(self):

        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = [data[0] for data in mini_batch]
        mcts_probs_batch = [data[1] for data in mini_batch]
        z_batch = [data[2] for data in mini_batch]
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)
        
        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step(
                    state_batch,
                    mcts_probs_batch,
                    z_batch,
                    self.learn_rate*self.lr_multiplier)
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs * (
                    np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)),
                    axis=1)
            )
            if kl > self.kl_targ * 4:
                break
        
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        # explained_var_old = (1 -
        #                      np.var(np.array(z_batch) - old_v.flatten()) /
        #                      np.var(np.array(z_batch)))
        # explained_var_new = (1 -
        #                      np.var(np.array(z_batch) - new_v.flatten()) /
        #                      np.var(np.array(z_batch)))
        
        print((
                # "kl:{:.5f},"
            #    "lr_multiplier:{:.3f},"
               "loss:{:.4f},"
            #    "entropy:{},"
            #    "explained_var_old:{:.3f},"
            #    "explained_var_new:{:.3f}"
               ).format(
                        # kl,
                        # self.lr_multiplier,
                        loss,
                        # entropy,
                        # explained_var_old,
                        # explained_var_new
                        ))
        return loss, entropy
    
    def loss(self, y_hat, y):
        return F.cross_entropy(y_hat, y)

    def policy_evaluate(self, testdl):

        good_preds = []
        reasoning_pred = []
        for i, batch in enumerate(testdl):
            batch = batch.to(self.device)
            d = self.process_batch(batch)
            prediction, moves = self.deduction.start_eval_deduction(d, self.mcts_agent)
            if prediction == 0 or prediction>20:
                prediction = 0
            good_preds.append(prediction == d.target)
            reasoning_pred.append(prediction != 0)
            print("i",i,"pred & y:", prediction, d.target, moves)

        win_ratio = sum(1*(good_preds)) / len(testdl)
        reasoning_ratio = sum(1*(reasoning_pred)) / len(testdl)
        
        print("   test acc: {}, reasoning ratio: {}".format(win_ratio, reasoning_ratio))
        return win_ratio

    def process_batch(self, batch):

        d = {}
        edge_type = batch.graphs.edge_attr.to(self.device)
        edge_index = batch.graphs.edge_index.to(self.device)

        _, edge_path_dict = get_rel_path((edge_index,edge_type), batch.queries[0])
        
        d['edge_path_dict'] = edge_path_dict
        d['query'] = batch.queries[0]
        d['target'] = int(batch.targets)

        d = torch_geometric.data.Data.from_dict(d)

        return d

    def preprocess(self):

        traindl = self.get_traindl()
        dataset = []
        order = []

        train_dl = []

        for i, batch in enumerate(traindl):

            train_dl.append(batch)
            
            d = self.process_batch(batch)
            dataset.append(d)

            # order.append(min([len(x) for x in edge_path_dict.values()]))
            order.append(max([len(x) for x in d.edge_path_dict.values()]))

        order = torch.tensor(order)
        _, order = torch.sort(order, dim=-1)

        return train_dl, dataset, order.tolist()

    def run(self):
        
        best_acc = 0.0

        PATH = 'model/best_policy_'+self.hparams.train_world+'.model'
        try:
            f = open('old_rules/rules_'+self.hparams.train_world+'.txt','r')
            a = f.read()
            rules = eval(a)
            f.close()

            f2 = open('old_rules/rules_score_'+self.hparams.train_world+'.txt','r')
            a2 = f2.read()
            rules_score = eval(a2)
            f2.close()

            self.print_confi_rules(rules, rules_score)

            self.deduction.set_rules(rules, rules_score)
            self.mcts_agent.mcts.set_rules(rules, rules_score)

            testdl = self.get_testdl()
            self.policy_value_net.load_state_dict(torch.load(PATH))
            print("\nNow Testing...\n")
            best_acc = self.policy_evaluate(testdl)
            print("Best test acc: %.4f" % best_acc)
        except FileNotFoundError:
            print("First training")
        except RuntimeError:
            print("Parameters changed")

        try:
            train_id = 0
            for n in range(self.hparams.epochs):
                print("Epoch", n, "start training...")
                train_dl, dataset, order = self.preprocess()
                testdl = self.get_testdl()
                self.data_buffer.clear()
                j = 0
                print(len(dataset))
                for i in order[train_id:]:
                    if n < 1:
                        batch = dataset[i]                        
                    else:
                        batch = dataset[j+train_id]
                    # print("\nepoch: {}, batch i:{}, target:{}".format(
                    #         n, j, batch.target))

                    self.collect_deduction_data(batch)

                    if (len(self.data_buffer) > self.batch_size) and (j % self.hparams.update_freq == 0):
                        loss, entropy = self.policy_update()
                    if j % (2*self.batch_size) == 0:
                        print("\nepoch: {}, batch i:{}\n".format(
                            n, j))
                        self.print_rules(self.deduction.rules)
                        self.print_rules_score(self.deduction.rules_score)

                    j += 1

                print("Training completed.")
                self.print_confi_rules(self.deduction.rules, self.deduction.rules_score)
                print("\nNow Testing...\n")
                tmp_acc = self.policy_evaluate(testdl)
                if tmp_acc > best_acc:
                    best_acc = tmp_acc
                    print("Best test acc: %.4f" % best_acc)
                    self.policy_value_net.save_model(PATH)

                    f = open('old_rules/rules_'+self.hparams.train_world+'.txt','w')
                    f.write(str(self.deduction.rules))
                    f.close()

                    f2 = open('old_rules/rules_score_'+self.hparams.train_world+'.txt','w')
                    f2.write(str(self.deduction.rules_score))
                    f2.close()
                                
        except KeyboardInterrupt:
            print('\n\rquit')

    def print_confi_rules(self, rules, rules_score):

        print("-------------\n"+"Printing Confident Rules...")
        mn = self.hparams.n_rel + self.hparams.n_unkn
        for body in rules:
            if body in rules_score and rules_score[body]>0:
                print(body//mn, body%mn,"->", rules[body], ", score:", rules_score[body])
        print()


    def print_rules(self, rules):
        print("-------------\n"+"Printing Rules...")
        mn = self.hparams.n_rel + self.hparams.n_unkn
        for body in rules:
            print(body//mn, body%mn,"->", rules[body], end=", ")
        print("\n")
    
    def print_rules_score(self, rules_score):
        print("-------------\n"+"Printing Rules Scores...")
        mn = self.hparams.n_rel + self.hparams.n_unkn
        for body in rules_score:
            print(body//mn, body%mn,"->", rules_score[body], end=", ")
        print("\n")

def load_config(name: str) -> Dict[str, Any]:
    """Load the config of a dataset
    Arguments:
        name {str} -- [description]
    Returns:
        Dict[str, Any] -- [description]
    """
    config_file_path = os.path.join(
        'data', name, "config.json"
    )
    config: Dict[str, Any] = json.load(open(config_file_path))
    return config

def load_rules(name: str) -> List[Any]:
    """Load the rules of a dataset
    Arguments:
        name {str} -- [description]
    Returns:
        Dict -- [description]
    """
    config: Dict[str, List[Any]] = load_config(name)
    return config["rules"]

def visual_rules(hparams):

    rules= load_rules(hparams.train_world)
    label2id_loc = os.path.join("data", "label2id.json")

    label2id = json.load(open(label2id_loc))
    rule_dict = {}

    n_edges = hparams.n_rel + hparams.n_unkn

    for i, d in enumerate(rules):
        if type(d['body']) == str:
            print(label2id[d['body']], '->', label2id[d['head']])
            continue
        # print rule
        for r in d['body']:
            print(label2id[r], end=" ")
        print("->",label2id[d['head']])

        # store index of the rule to dict
        key = label2id[d['body'][0]] * n_edges + label2id[d['body'][1]]
        rule_dict[key] = label2id[d['head']]
    
    return rule_dict

if __name__ == '__main__':

    torch.multiprocessing.set_sharing_strategy('file_system')

    parser = ArgumentParser(add_help=False)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--train_world", default="rule_17", type=str)
    parser.add_argument("--n_rel", default=21, type=int, help="20 classes including UNK rel")
    parser.add_argument("--n_unkn", default=50, type=int)

    parser.add_argument("--episode_decay", default=0.003, type=float)
    parser.add_argument("--learning_rate", default=0.01, type=float)
    parser.add_argument("--lr_multiplier", default=1.0, type=float)
    parser.add_argument("--temp", default=1.0, type=float)
    parser.add_argument("--kl_targ", default=0.02, type=float)
    parser.add_argument("--n_playout", default=200, type=int)
    parser.add_argument("--c_puct", default=10, type=int)
    parser.add_argument("--buffer_size", default=5000, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--update_freq", default=16, type=int)

    parser.add_argument("--simple_true_score", default=1.0, type=float)
    parser.add_argument("--wrong_allowance", default=0.5, type=float)

    parser.add_argument("--v0", default=0.6, type=float)
    parser.add_argument("--v1", default=0.3, type=float)
    parser.add_argument("--v2", default=-0.05, type=float)
    parser.add_argument("--v3", default=-0.1, type=float)
    parser.add_argument("--v4", default=-0.3, type=float)

    # parser.add_argument("--v0", default=0.5, type=float)
    # parser.add_argument("--v1", default=0.5, type=float)
    # parser.add_argument("--v2", default=0.5, type=float)
    # parser.add_argument("--v3", default=0.5, type=float)
    # parser.add_argument("--v4", default=0.5, type=float)

    parser.add_argument("--v_T_pos", default=0.1, type=float)
    parser.add_argument("--v_T_neg", default=-1.0, type=float)
    parser.add_argument("--bad_rule_thres", default=-1.2, type=float)

    hyperparams = parser.parse_args()
    visual_rules(hyperparams)

    trainer = Trainer(hyperparams)
    trainer.run()