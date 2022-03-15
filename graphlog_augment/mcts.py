import numpy as np
import copy
import random

def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

class TreeNode():
    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p

    def expand(self, action_priors):
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        self._n_visits += 1
        self._Q += 1.0*(leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        return self._children == {}

    def is_root(self):
        return self._parent is None

class MCTS(object):

    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000):
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout
        self.rules = {}
        self.rules_score = {}

    def set_rules(self, rules, rules_score):
        self.rules = rules 
        self.rules_score = rules_score

    def _playout(self, state, in_eval=False):

        node = self._root
        while(1):
            if node.is_leaf():
                break
            action, node = node.select(self._c_puct)
            if action not in state.action_dict.keys():
                node = node._parent
                break
            predicate  = self.get_pred(action, state.width, in_eval)
            if predicate == 0:
                return 
            else:
                state.do_move(action, predicate)

        end, rule_len = state.is_end()

        if not end:
            action_probs, leaf_value = self._policy(state, self.rules, self.rules_score)

            node.expand(action_probs)
            bad_node = set(node._children.keys()) - set(state.availables)
            for bn in bad_node:
                del node._children[bn]
        else:
            leaf_value = 1.0/rule_len

        # Update value and visit count of nodes in this traversal.
        node.update_recursive(-leaf_value)

    def del_inves(self, inves, width):
        actions = list(self.rules.keys())
        for key in actions:
            if key in self.rules and self.rules[key] == inves:
                # print(">>>>>> del", key, "->", self.rules[key])
                self.rm_rule(key,width)

    def get_pred(self, move, width, in_eval=False):

        if self.rules and move in self.rules.keys():
            pred = self.rules[move]
        else:
            if in_eval:
                return 0
            pred_buffer = set(range(21,width)) - set(self.rules.values())
            # print(">>> buffer", pred_buffer)
            try:
                pred = np.random.choice(np.array(list(pred_buffer)))
                # print(">>>>>> occupied:", pred)
            except ValueError:
                # print("+++++++\nfully occupied\n++++++++")
                worst = sorted(self.rules_score.values())[:2]
                actions = list(self.rules_score.keys())
                # print(worst)
                if len(self.rules_score) < 1 or max(worst)>=0:
                    inves = random.choice(range(21,width))
                    self.del_inves(inves, width)
                    return self.get_pred(move, width)

                for key in actions:
                    if self.rules_score[key] <= max(worst) and max(worst)<0:
                        if key in self.rules:
                            # print(">>>>>> del", key, "->", self.rules[key],"score", self.rules_score[key])
                            self.rm_rule(key, width)
                        if key in self.rules_score:
                            del self.rules_score[key]
                return self.get_pred(move, width)
                
            self.rules[move] = pred
        return pred

    def rm_rule(self, move, width):
        
        if move in self.rules:
            inves = int(self.rules[move])
            del self.rules[move]
            
            actions = list(self.rules.keys())
            if inves > 20:
                # agent.mcts._policy.act_fc1.weight[:,inves].data.fill_(0.0)
                for m in actions:
                    if (m%width == inves or m//width == inves) and m in self.rules:
                        # print(">>>>>> del", m, "->", self.rules[m])
                        self.rm_rule(m, width)

    def update_rules(self, rules, rules_score):
        self.rules = rules
        self.rules_score = rules_score

    def get_move_probs(self, state, temp=1e-3, in_eval=False):

        # print("get_move_probs", state.edge_path_dict)
        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy, in_eval)
        
        act_visits = [(act, node._n_visits)
                      for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))

        return acts, act_probs

    def update_with_move(self, last_move):

        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)


class MCTSAgent(object):

    def __init__(self, policy_value_function,
                 c_puct=5, n_playout=2000):
        self.mcts = MCTS(policy_value_function, c_puct, n_playout)

    def set_agent_ind(self, p):
        self.agent = p

    def reset_agent(self):
        self.mcts.update_with_move(-1)

    def get_action(self, allpair, temp=1e-3, return_prob=0, in_eval=False):

        sensible_moves = allpair.availables

        # the pi vector returned by MCTS as in the alphaGo Zero paper
        move_probs = np.zeros(allpair.width*allpair.width)

        if len(sensible_moves) > 0:
            acts, probs = self.mcts.get_move_probs(allpair, temp, in_eval)
            move_probs[list(acts)] = probs
            if not in_eval:
                move = np.random.choice(
                    acts,
                    p=0.9*probs + 0.1*np.random.dirichlet(0.3*np.ones(len(probs)))
                )
                self.mcts.update_with_move(move)
            elif in_eval:
                move = np.random.choice(acts, p=probs)
                self.mcts.update_with_move(-1)
            else:
                pass
            
            if return_prob:
                return move, move_probs
            else:
                return move





