import os
import random
import numpy as np
from argparse import ArgumentParser
from collections import defaultdict, deque

import copy
import json
import os
from typing import Any, Dict, List, Optional, Set, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torch_geometric.data import Batch as GeometricBatch
from torch_geometric.data import Data as GeometricData

from inits import *
from dedution import Allpair, Deduction
from mcts import MCTSAgent
from policy_value_net import PolicyValueNet

from base import Fact, Data, Instance
from evaluation import accuracy

# import certifi
# import urllib3
# http = urllib3.PoolManager(
#      cert_reqs='CERT_REQUIRED',
#      ca_certs=certifi.where()
# )

class Trainer():
    def __init__(self, hparams): 

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

        self.get_rel_dict()

    def get_rel_dict(self):
        f = open('rel_dict','r')
        a = f.read()
        self.rel_dict = eval(a)
        f.close()

    def get_traindl_testdl(self):

        train_path = self.hparams.train
        test_paths = self.hparams.test

        data = Data(train_path=train_path, test_paths=test_paths)

        relation_to_predicate = data.relation_to_predicate
        predicate_to_relations = data.predicate_to_relations
        entity_lst, predicate_lst, relation_lst = data.entity_lst, data.predicate_lst, data.relation_lst

        nb_examples = len(data.train)
        nb_entities = len(entity_lst)
        nb_relations = len(relation_lst)

        entity_to_idx = {e: i for i, e in enumerate(entity_lst)}
        relation_to_idx = {r: i+1 for i, r in enumerate(relation_lst)}

        self.nodeid2entity = {entity_to_idx[e]:e for e in entity_to_idx}

        train_dl = []
        for i, d in enumerate(data.train):
            B = d.story
            Q = d.target

            edge_index_l = []
            edge_index_r = []
            edge_type = []
            batch = {}

            for m in B: 
                edge_index_l.append(entity_to_idx[m[0]])
                edge_index_r.append(entity_to_idx[m[2]])
                edge_type.append(relation_to_idx[m[1]])

            edge_index = torch.cat([torch.tensor(edge_index_l).view(1,-1), torch.tensor(edge_index_r).view(1,-1)], dim=0).to(self.device)
            edge_type = torch.tensor(edge_type).to(self.device)

            query = (entity_to_idx[Q[0]], entity_to_idx[Q[2]])
            target = relation_to_idx[Q[1]]

            batch['edge_index'] = edge_index
            batch['edge_type'] = edge_type
            batch['query'] = query
            batch['target'] = target

            batch = torch_geometric.data.Data.from_dict(batch)
            train_dl.append(batch)

        test_dl = []
        for j, test_raw in enumerate(test_paths):
            test_d = []
            for i, d in enumerate(data.test[test_raw]):
                B = d.story
                Q = d.target

                edge_index_l = []
                edge_index_r = []
                edge_type = []
                batch = {}

                for m in B: 
                    edge_index_l.append(entity_to_idx[m[0]])
                    edge_index_r.append(entity_to_idx[m[2]])
                    edge_type.append(relation_to_idx[m[1]])

                edge_index = torch.cat([torch.tensor(edge_index_l).view(1,-1), torch.tensor(edge_index_r).view(1,-1)], dim=0).to(self.device)
                edge_type = torch.tensor(edge_type).to(self.device)

                query = (entity_to_idx[Q[0]], entity_to_idx[Q[2]])
                target = relation_to_idx[Q[1]]

                batch['edge_index'] = edge_index
                batch['edge_type'] = edge_type
                batch['query'] = query
                batch['target'] = target

                batch = torch_geometric.data.Data.from_dict(batch)
                test_d.append(batch)
            test_dl.append(test_d)
        
        return train_dl,test_dl

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
            d = self.process_batch(batch, with_node=True)
            prediction, moves, reasoning = self.deduction.start_eval_deduction(d, self.mcts_agent, self.rel_dict, self.nodeid2entity)
            if prediction == 0 or prediction>self.hparams.n_rel-1:
                prediction = 0
            good_preds.append(prediction == d.target)
            reasoning_pred.append(prediction != 0)
            if self.hparams.full_reason:
                print("i",i,"pred & y:", prediction, d.target, reasoning)
            else:
                print("i",i,"pred & y:", prediction, d.target, moves)

        win_ratio = sum(1*(good_preds)) / len(testdl)
        reasoning_ratio = sum(1*(reasoning_pred)) / len(testdl)
        
        print("   test acc: {}, reasoning ratio: {}".format(win_ratio, reasoning_ratio))
        return win_ratio
    
    def process_batch(self, batch, with_node=False):

        d = {}
        edge_type = batch.edge_type.to(self.device)
        edge_index = batch.edge_index.to(self.device)

        node_path_dict, edge_path_dict = get_rel_path((edge_index,edge_type), batch.query)

        if with_node:
            d['node_path_dict'] = node_path_dict
        d['edge_path_dict'] = edge_path_dict
        d['query'] = batch.query
        d['target'] = int(batch.target)

        d = torch_geometric.data.Data.from_dict(d)

        return d

    def preprocess(self):

        traindl, test_dl = self.get_traindl_testdl()
        dataset = []
        order = []

        for i, batch in enumerate(traindl):
            
            d = self.process_batch(batch)
            dataset.append(d)

            # order.append(min([len(x) for x in edge_path_dict.values()]))
            order.append(max([len(x) for x in d.edge_path_dict.values()]))

        order = torch.tensor(order)
        _, order = torch.sort(order, dim=-1)

        return dataset, order.tolist(), test_dl

    def run(self):
        
        best_acc = 0.0

        PATH = 'model/best_policy_'+self.hparams.log_num+'.model'
        try:
            f = open('old_rules/rules_'+self.hparams.log_num+'.txt','r')
            a = f.read()
            rules = eval(a)
            f.close()

            f2 = open('old_rules/rules_score_'+self.hparams.log_num+'.txt','r')
            a2 = f2.read()
            rules_score = eval(a2)
            f2.close()

            self.print_confi_rules(rules, rules_score)

            self.deduction.set_rules(rules, rules_score)
            self.mcts_agent.mcts.set_rules(rules, rules_score)

            dataset, order, testdl = self.preprocess()
            self.policy_value_net.load_state_dict(torch.load(PATH))
            print("\nNow Testing...\n")
            print(len(testdl))
            if type(best_acc) is float:
                best_acc = [0.0]*len(testdl)
            for m in range(len(testdl)):
                best_acc[m] = self.policy_evaluate(testdl[m])
                print("Best test acc of testset", m, ": %.4f" % best_acc[m], "\n")
        except FileNotFoundError:
            print("First training")
        except RuntimeError:
            print("Parameters changed")

        try:
            train_id = 0
            for n in range(self.hparams.epochs):
                print("Epoch", n, "start training...")
                dataset, order, testdl = self.preprocess()
                print(len(order), len(testdl))
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
                    if j % (8*self.batch_size) == 0:
                        print("\nepoch: {}, batch i:{}\n".format(
                            n, j))
                        self.print_rules(self.deduction.rules)
                        self.print_rules_score(self.deduction.rules_score)

                    j += 1

                print("Training completed.")
                self.print_confi_rules(self.deduction.rules, self.deduction.rules_score)
                print("\nNow Testing...\n")
                tmp_acc = [0.0] * len(testdl)
                if type(best_acc) is float:
                    best_acc = [0.0]*len(testdl)

                for m in range(len(testdl)):
                    print("testset", m)
                    tmp_acc[m] = self.policy_evaluate(testdl[m])

                    if tmp_acc[m] > best_acc[m]:
                        best_acc[m] = tmp_acc[m]
                        print("Best test acc of testset", m, ": %.4f" % best_acc[m],"\n")
                        self.policy_value_net.save_model(PATH)

                        f = open('old_rules/rules_'+self.hparams.log_num+'.txt','w')
                        f.write(str(self.deduction.rules))
                        f.close()

                        f2 = open('old_rules/rules_score_'+self.hparams.log_num+'.txt','w')
                        f2.write(str(self.deduction.rules_score))
                        f2.close()
                                
        except KeyboardInterrupt:
            print('\n\rquit')

    def print_confi_rules(self, rules, rules_score):

        print("-------------\n"+"Printing Confident Rules...")
        mn = self.hparams.n_rel + self.hparams.n_unkn
        for body in rules:
            A = self.rel_dict[body//mn] if body//mn in self.rel_dict else "unknown_"+str(body//mn)
            B = self.rel_dict[body%mn] if body%mn in self.rel_dict else "unknown_"+str(body%mn)
            C = self.rel_dict[rules[body]] if rules[body] in self.rel_dict else "unknown_"+str(rules[body])
            if body in rules_score and rules_score[body]>0:
                print("(",A, ",", B,")","->", C, ", score:", rules_score[body])
        print()

    def print_rules(self, rules):
        print("-------------\n"+"Printing Rules...")
        mn = self.hparams.n_rel + self.hparams.n_unkn
        for body in rules:
            A = self.rel_dict[body//mn] if body//mn in self.rel_dict else "unknown_"+str(body//mn)
            B = self.rel_dict[body%mn] if body%mn in self.rel_dict else "unknown_"+str(body%mn)
            C = self.rel_dict[rules[body]] if rules[body] in self.rel_dict else "unknown_"+str(rules[body])
            print("(",A , ",", B,")","->", C, end=", ")
        print("\n")
    
    def print_rules_score(self, rules_score):
        print("-------------\n"+"Printing Rules Scores...")
        mn = self.hparams.n_rel + self.hparams.n_unkn
        for body in rules_score:
            A = self.rel_dict[body//mn] if body//mn in self.rel_dict else "unknown_"+str(body//mn)
            B = self.rel_dict[body%mn] if body%mn in self.rel_dict else "unknown_"+str(body%mn)
            print("(",A, ",", B,")","->", rules_score[body], end=", ")
        print("\n")

if __name__ == '__main__':

    torch.multiprocessing.set_sharing_strategy('file_system')

    parser = ArgumentParser(add_help=False)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--log_num", default='0', type=str)
    parser.add_argument('--train', action='store', type=str)
    parser.add_argument('--test', nargs='+', type=str)
    parser.add_argument("--n_rel", default=23, type=int, help="22 classes including UNK rel")
    parser.add_argument("--n_unkn", default=27, type=int)

    parser.add_argument("--episode_decay", default=0.003, type=float)
    parser.add_argument("--learning_rate", default=0.01, type=float)
    parser.add_argument("--lr_multiplier", default=1.0, type=float)
    parser.add_argument("--temp", default=1.0, type=float)
    parser.add_argument("--kl_targ", default=0.02, type=float)
    parser.add_argument("--n_playout", default=200, type=int)
    parser.add_argument("--c_puct", default=10, type=int)
    parser.add_argument("--buffer_size", default=5000, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--update_freq", default=128, type=int)

    parser.add_argument("--simple_true_score", default=10.0, type=float)
    # parser.add_argument("--wrong_allowance", default=-5.0, type=float)

    parser.add_argument("--v0", default=0.6, type=float)
    parser.add_argument("--v1", default=0.3, type=float)
    parser.add_argument("--v2", default=-0.05, type=float)
    parser.add_argument("--v3", default=-0.1, type=float)
    parser.add_argument("--v4", default=-0.3, type=float)

    parser.add_argument("--v_T_pos", default=0.1, type=float)
    parser.add_argument("--v_T_neg", default=-1.0, type=float)
    parser.add_argument("--bad_rule_thres", default=-1.2, type=float)

    parser.add_argument("--full_reason", default=False, type=bool)

    hyperparams = parser.parse_args()

    trainer = Trainer(hyperparams)
    trainer.run()