import numpy as np
import torch
import copy

class Allpair():

    def __init__(self, width):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.width = width
        self.num_features = 8
        self.rule_len = 0

    def init_allpair(self, batch):

        self.edge_path_dict = batch.edge_path_dict
        # keep available moves in a list
        self.availables = set()
        self.action_dict = {}

        for i, ep in enumerate(self.edge_path_dict.values()):
            for j in range(len(ep)-1):

                key = int(ep[j]*self.width + ep[j+1])
                self.availables.add(key)
                if key not in self.action_dict.keys():
                    self.action_dict[key] = set([i])
                else:
                    self.action_dict[key].add(i)

        self.availables = list(self.availables)

        self.last_move = -1
        self.rule_len = 0

    def current_state(self, rules, rules_score):

        square_state = np.zeros((self.num_features, self.width, self.width))
        
        stats = []
        for n, ep in enumerate(self.edge_path_dict.values()):
            eep = []
            for i in range(len(ep)-1):
                eep.append(ep[i] * self.width + ep[i+1]) 
            stats.append(torch.tensor(eep).to(self.device))
        
        pred_list = torch.nn.utils.rnn.pad_sequence(stats, batch_first=True, padding_value=-1)
        unique_preds, inv_indices, counts = torch.unique(pred_list, return_counts=True, return_inverse=True)
        
        is_single_path = False
        if unique_preds[0] < 0:
            unique_preds = unique_preds[1:]
            is_single_path = True

        for j, pred in enumerate(unique_preds):
            l = int(pred // self.width)
            r = int(pred % self.width)
            
            # number of occurence
            t = j+1 if is_single_path else j

            square_state[0][l,r] = int(counts[t])
            # max freq location and the occurence at the location
            square_state[1][l,r], square_state[2][l,r] = (int(x) for x in torch.topk(torch.sum(inv_indices==t, dim=0), 1))
            square_state[3][l,r] = int(int(pred) in rules)
            square_state[4][l,r] = int(rules[int(pred)]<21) if int(pred) in rules else 0
            square_state[5][l,r] = rules_score[int(pred)] if int(pred) in rules_score else 0
            square_state[6][l,r] = l
            square_state[7][l,r] = r
        
        return square_state

    def do_move(self, move, pred):
        
        pred = torch.tensor([pred]).to(self.device)
        path_to_remove = []

        all_path_ids = list(self.edge_path_dict.keys())
        
        for path_id in all_path_ids:
            if move not in self.action_dict:
                continue
            if path_id not in self.action_dict[move]:
                del self.edge_path_dict[path_id]
            else: 
                p_len = len(self.edge_path_dict[path_id])

                possible_action_path = []
                for i in range(p_len-1):
                    possible_action_path.append(int(self.edge_path_dict[path_id][i]*self.width+self.edge_path_dict[path_id][i+1]))

                occurence_action = np.where(np.array(possible_action_path)==move)[0]
                selection = np.random.choice(occurence_action)
                # selection = occurence_action[0]

                self.edge_path_dict[path_id] = torch.cat([self.edge_path_dict[path_id][:selection], pred, self.edge_path_dict[path_id][selection+2:]])
        
        # keep available moves in a list
        self.availables = set()
        self.action_dict = {}

        for i in self.edge_path_dict.keys():
            ep = self.edge_path_dict[i]
            for j in range(len(ep)-1):

                key = int(ep[j]*self.width + ep[j+1])
                self.availables.add(key)
                if key not in self.action_dict.keys():
                    self.action_dict[key] = set([i])
                else:
                    self.action_dict[key].add(i)

        self.availables = list(self.availables)

        self.last_move = move
        self.rule_len += 1

    def is_end(self):

        path_lens  = [len(x) for x in self.edge_path_dict.values()]

        if min(path_lens) == 1:
            return True, self.rule_len
        return False, -1

class Deduction():

    def __init__(self,rules,allpair,hparams):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.allpair = allpair
        self.rules = rules
        self.rules_score = {}
        self.hparams = hparams
        self.decay = self.hparams.episode_decay

    def clear_rules(self):
        self.rules = {}
        self.rules_score = {}

    def set_rules(self, rules, rules_score):
        self.rules = rules 
        self.rules_score = rules_score

    def start_eval_deduction(self, batch, mcts_agent):

        self.allpair.init_allpair(batch)
        moves = []

        while True:

            move, move_probs = mcts_agent.get_action(self.allpair, return_prob=1, in_eval=True)
            # print("move", move, np.arange(len(move_probs))[move_probs>0], np.array(move_probs)[move_probs>0], end=" ")

            if move in mcts_agent.mcts.rules.keys():
                self.allpair.do_move(move, mcts_agent.mcts.rules[move])
                moves.append(str(move//self.allpair.width)+" "+str(move%self.allpair.width)+" -> "+str(mcts_agent.mcts.rules[move]))
            else: 
                moves.append(str(move//self.allpair.width)+" "+str(move%self.allpair.width)+" -> 0")
                return 0, moves

            end, _ = self.allpair.is_end()
            if end:
                if move in mcts_agent.mcts.rules.keys():
                    return mcts_agent.mcts.rules[move], moves
                else: 
                    return 0, moves

    def update_investigate(self, investigate, target):

        actions = list(self.rules.keys())

        for action in actions:
            l = action//self.allpair.width
            r = action%self.allpair.width
            pred_id = self.rules[action]
            if l == investigate and r == investigate:
                key = target*self.allpair.width+target
            elif l == investigate:
                key = target*self.allpair.width+r
            elif r == investigate:
                key = l*self.allpair.width+target
            else:
                continue
            if key not in actions or self.rules[key]>20:
                # if key//self.allpair.width<21 and key%self.allpair.width<21:
                    # print("-----\n+++++++++++++++++\n>>> From", l,r, "->",pred_id, end=" ")
                    # print("... To", key//self.allpair.width,key%self.allpair.width, "->", pred_id, end="\n+++++++++++++++++\n")
                # print(">>>>>>>>>freed",action)
                del self.rules[action]
                self.rules[key] = pred_id
                # print("\n>>>> WOW freed", investigate, action, "->", key)
                # print(self.rules)
                # exit(0)
            else:
                # print(">>>>>>>>>freed",action)
                del self.rules[action]

    def update_rules(self, move, target):

        # old head in the rule memory
        investigate = int(self.rules[move])   

        if investigate < self.hparams.n_rel:
            if move in self.rules_score.keys() and investigate != target:
                if self.rules_score[move] < 0:
                    self.rules[move] = target
                    self.rules_score[move] = 0
            pass
        else: 
            self.rules[move] = target
            actions = list(self.rules.keys())
            for key in actions:
                if key in self.rules and self.rules[key] == investigate:
                    # print(">>>>>> del", key, "->", self.rules[key])
                    self.rules[key] = target
                    self.rules_score[key] = 0

            self.update_investigate(investigate, target)

    def rm_rule(self, move):
        
        if move in self.rules:
            inves = int(self.rules[move])
            del self.rules[move]
            
            actions = list(self.rules.keys())
            if inves > 20:
                # mcts_agent.mcts._policy.act_fc1.weight[:,inves].data.fill_(0.0)
                for m in actions:
                    if m%self.allpair.width == inves or m//self.allpair.width == inves:
                        # print(">>>>>> del", m, "->", self.rules[m])
                        self.rm_rule(m)

    def del_inves(self, inves):
        actions = list(self.rules.keys())
        for key in actions:
            if key in self.rules and self.rules[key] == inves:
                # print(">>>>>> del", key, "->", self.rules[key])
                self.rm_rule(key)

    def start_deduction(self, batch, mcts_agent, is_shown=0, temp=1e-3):
        self.allpair.init_allpair(batch)
        states, mcts_probs, found_in_rules, found_ground_rules, found_in_tgt_rules = [], [], [], [], []

        poten_moves = []
        moves = []

        mcts_agent.mcts.update_rules(copy.deepcopy(self.rules), copy.deepcopy(self.rules_score))

        while True:
            move, move_probs = mcts_agent.get_action(copy.deepcopy(self.allpair),
                                                 temp=temp,
                                                 return_prob=1)

            # print("move", move, np.arange(len(move_probs))[move_probs>0], np.array(move_probs)[move_probs>0], end=" ")
            
            # store the data
            states.append(self.allpair.current_state(mcts_agent.mcts.rules, self.rules_score))
            mcts_probs.append(move_probs)
            found_in_rules.append(move in self.rules)
            found_ground_rules.append(move in self.rules and self.rules[move] < self.hparams.n_rel and move//self.allpair.width<self.hparams.n_rel and move%self.allpair.width<self.hparams.n_rel) 
            found_in_tgt_rules.append(move in self.rules and self.rules[move] < self.hparams.n_rel) 
            poten_moves.append(move not in self.rules.keys() and move//self.allpair.width<self.hparams.n_rel and move%self.allpair.width<self.hparams.n_rel)

            moves.append(move)
            
            # print(self.allpair.edge_path_dict)
            
            # perform a move
            if move in mcts_agent.mcts.rules.keys():
                predicate = mcts_agent.mcts.rules[move]
            elif move in self.rules.keys():
                predicate = self.rules[move]
            else:
                predicate = mcts_agent.mcts.get_pred(move,self.allpair.width)
            self.allpair.do_move(move, predicate)

            if move not in self.rules.keys():
                values = list(self.rules.values())
                inves = mcts_agent.mcts.rules[move]
                if inves > self.hparams.n_rel-1 and inves in values:
                    self.del_inves(inves)
                self.rules[move] = mcts_agent.mcts.rules[move]
            else:
                old_head = self.rules[move]
                if old_head != predicate:
                    if old_head > self.hparams.n_rel-1:
                        self.update_rules(move, predicate)
            
            end, rule_len = self.allpair.is_end()
            if end:
                
                self.update_rules(move, batch.target)
                # print("<<< Global rules ", self.rules)

                z = np.zeros(len(found_in_rules))
                scores = np.zeros(len(found_in_rules))

                z[:] = -0.02

                scores[:] = self.hparams.v4
                scores[np.array(found_in_rules) == True] = self.hparams.v3
                scores[np.array(found_in_tgt_rules) == True] = self.hparams.v1
                scores[np.array(found_ground_rules) == True] = self.hparams.v0
                scores[np.array(poten_moves) == True] = self.hparams.v2
                
                # print(">>> prediction", mcts_agent.mcts.rules[move])
                
                # if last move gives prediction
                if mcts_agent.mcts.rules[move] == int(batch.target):
                    scores[:] += self.hparams.v_T_pos
                    if len(found_in_rules) == 1:
                        scores[:] += self.hparams.simple_true_score
                    z[:] = 1
                elif mcts_agent.mcts.rules[move] < self.hparams.n_rel:
                    scores[:] = self.hparams.v_T_neg
                    if len(found_in_rules) == 1:
                        scores[:] += self.hparams.wrong_allowance
                    z[:] = -1
                
                # if min(scores[:-1])>0:
                #     scores[:-1] = 1

                for n, m in enumerate(moves):

                    if m not in self.rules_score:
                        self.rules_score[m] = 0.0
                    self.rules_score[m] += scores[n]

                actions = list(self.rules_score.keys())
                for m in actions:
                    if m not in self.rules:
                        del self.rules_score[m]
                        continue
                    elif self.rules_score[m]>0:
                        self.rules_score[m] *= (1-self.decay)
                    elif self.rules_score[m]<0:
                        self.rules_score[m] *= (1+self.decay)
                    self.rules_score[m] = round(self.rules_score[m], 4)

                actions = list(self.rules_score.keys())
                for m in actions:
                    if m not in self.rules:
                        del self.rules_score[m]
                        continue
                    elif self.rules_score[m]>0:
                        self.rules_score[m] *= (1-self.decay)
                    elif self.rules_score[m]<0:
                        self.rules_score[m] *= (1+self.decay)
                    self.rules_score[m] = round(self.rules_score[m], 4)

                actions = list(self.rules_score.keys())
                for m in actions:
                    if self.rules_score[m] < self.hparams.bad_rule_thres:
                        try:
                            # print(">>>>>> del", m, "->", self.rules[m],"score", self.rules_score[m])
                            self.rm_rule(m)
                        except KeyError:
                            # print("Already removed.", m)
                            pass
                        del self.rules_score[m]

                # print("------------\n", self.rules_score)
                
                # reset MCTS root node
                mcts_agent.reset_agent()

                return rule_len, zip(states, mcts_probs, z)
