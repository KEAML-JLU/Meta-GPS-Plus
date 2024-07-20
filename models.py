import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from kmeans_pytorch import kmeans
from sklearn.metrics import f1_score
from learner import Learner, Scaling, Translation, Transform
from utils import sgc_precompute

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Meta(nn.Module):
    def __init__(self, args, config, config_transform, config_scal, config_trans, feat, label_num, adj, adj_tilde, adj_two, id_by_class):
        super(Meta, self).__init__()
        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.scl_lr = args.scl_lr
        self.n_way = args.way
        self.k_spt = args.shot
        self.k_qry = args.qry
        self.task_num = args.task_num
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        self.feat = feat
        self.id = id_by_class
        self.deg = args.deg
        self.k = args.k
        self.adj = adj
        self.adj_tilde = adj_tilde # one-hop adj
        self.adj_two = adj_two # two-hop adj

        self.MI = True  
        self.hidden = args.hidden
        self.mlp = MLP(self.hidden)
        fc_params = nn.Linear(self.hidden, self.n_way, bias=None)
        self.fc = [fc_params.weight.detach()] * self.task_num
        for i in range(self.task_num): self.fc[i].requires_grad = True

        self.net = Learner(config)
        self.net = self.net.to(device)

        self.scaling = Scaling(config_scal, args, label_num)
        self.scaling = self.scaling.to(device)

        self.translation = Translation(config_trans, args, label_num)
        self.translation = self.translation.to(device)
        
        self.transformation = Transform(config_transform)
        self.transformation = self.transformation.to(device)
        
        
        self.alpha = args.alpha
        self.scl = SCL(self.hidden, self.hidden, args.tem)
        self.scl = self.scl.to(device)
        
        self.beta = args.beta

        self.meta_optim = optim.Adam([{'params':self.net.parameters()}, {'params':self.mlp.trans.parameters()},
                                      {'params':self.scaling.parameters()}, {'params':self.scl.parameters(), 'lr':self.scl_lr},{'params':self.translation.parameters()},{'params':self.transformation.parameters()}], lr=self.meta_lr)
        
    def reset_fc(self):
        self.fc = [torch.Tensor(self.n_way, self.hidden)]*self.task_num

    def prework(self, meta_information):
        return self.mlp(meta_information)

    def preforward(self, support, fc):
        return F.linear(support, fc, bias=None)

    def forward(self, x_spt, y_spt, x_qry, y_qry, meta_information_dict, class_selected, labels, training):

        self.h = self.transformation(self.feat) # [25000,16]
        self.h = sgc_precompute(self.h, self.adj_tilde, self.adj_two)
        self.id_by_class_prototype_embedding = {k: self.h[np.array(self.id[k])].mean(0) for k in self.id.keys()}
        step = self.update_step if training is True else self.update_step_test
        querysz = self.n_way * self.k_qry
        losses_s = [0 for _ in range(step)]
        losses_q = [0 for _ in range(step + 1)]  # losses_q[i] is the loss on step i
        corrects = [0 for _ in range(step + 1)]
        f1s = [0 for _ in range(step + 1)]
        meta_information_dict = {}
        for i in range(self.task_num):
            meta_information_dict[i] = torch.stack([self.id_by_class_prototype_embedding[int(k)] for k in class_selected[i]]).to(device)

        for i in range(self.task_num):
            meta_information = meta_information_dict[i] # [n_way, hidden]
            self.fc[i] = self.prework(meta_information) # [n_way, hidden]
            logits_two = self.preforward(self.h[x_spt[i]], self.fc[i]) # the meta information of x_support
            logits_three = self.preforward(self.h[x_qry[i]], self.fc[i]) # the meta information of x_query
            
            # for scl training
            meta_label = torch.flip(torch.unique(y_spt[i], sorted=False), dims=[0])
            label = torch.cat([y_spt[i], y_qry[i], meta_label])
            fea_scl = torch.cat([self.h[x_spt[i]], self.h[x_qry[i]], meta_information], dim=0)
            
            # for self training
            unselect = np.array([])
            combine = np.hstack((x_spt[i], x_qry[i]))
            
            for cla in class_selected[i]:
                unselect = np.union1d(unselect, np.array(self.id[cla]))
            unselect = np.setdiff1d(unselect, combine)
            
            # use student t distribution
            q = 1.0 / (1.0 + torch.sum(torch.pow(self.h[unselect].unsqueeze(1) - meta_information, 2), 2) / self.deg)
            q = q.pow((self.deg + 1.0) / 2.0)
            q = (q.t() / torch.sum(q, 1)).t()
            q = F.softmax(q, dim=-1)
            select = []
            #plan one
            _, index = q.topk(self.k, dim=0)
            for k in range(self.n_way):
                select.append(q[index[:,k]])
            
            q = torch.cat(select)
            p = q**2 / q.sum(0)
            p = (p.t() / p.sum(1)).t()
            
            logits_value = self.net(logits_two, vars=None)#[x_spt[i]] # logits_value is intermediate variable
            
            scaling = self.scaling(logits_value)
            translation = self.translation(logits_value)
            adapted_prior = []
            for s in range(len(scaling)):
                adapted_prior.append(torch.mul(self.net.parameters()[s], (scaling[s] + 1)) + translation[s])
            logits = self.net(logits_two, adapted_prior)

            loss = F.cross_entropy(logits, y_spt[i]) #+ (h_theta_update - h_theta) * 0.001
            losses_s[0] += loss
            grad = torch.autograd.grad(loss, adapted_prior)
            fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, adapted_prior)))

            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net(logits_three, adapted_prior)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                acc_q = torch.eq(pred_q, y_qry[i]).sum().item()
                
                f1_q = f1_score(y_qry[i].cpu(), pred_q.cpu(), average='weighted', labels=np.unique(pred_q.cpu()))
                losses_q[0] += loss_q
                corrects[0] = corrects[0] + acc_q
                f1s[0] = f1s[0] + f1_q

            # this is the loss and accuracy after the first update
            with torch.no_grad():
                logits_q = self.net(logits_three, fast_weights)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                acc_q = torch.eq(pred_q, y_qry[i]).sum().item()
                f1_q = f1_score(y_qry[i].cpu(), pred_q.cpu(), average='weighted', labels=np.unique(pred_q.cpu()))
                losses_q[1] += loss_q
                corrects[1] = corrects[1] + acc_q
                f1s[1] = f1s[1] + f1_q

            for k in range(1, step):
                # 1. run the i-th task and compute loss for k=1~K-1
                logits = self.net(logits_two, fast_weights)
                loss = F.cross_entropy(logits, y_spt[i])
                losses_s[k] += loss
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, fast_weights)
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(map(lambda p: p[1] - self.update_lr * p[0], zip(grad, fast_weights)))
                # this is modify
                logits_q = self.net(logits_three, fast_weights)
                loss_q = F.cross_entropy(logits_q, y_qry[i])
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                acc_q = torch.eq(pred_q, y_qry[i]).sum().item()
                f1_q = f1_score(y_qry[i].cpu(), pred_q.cpu(), average='weighted', labels=np.unique(pred_q.cpu()))

                if training == True:
                    l2_loss = torch.sum(torch.stack([torch.norm(k) for k in scaling]))
                    l2_loss += torch.sum(torch.stack([torch.norm(k) for k in translation]))
                    l2_loss = l2_loss * 0.0001
                    scl_loss = self.scl(fea_scl, label) * self.alpha
                    kl_loss = F.kl_div(torch.log(q), p, reduction='batchmean') * self.beta

                    losses_q[k + 1] += (loss_q + l2_loss + scl_loss + kl_loss)
                else:
                    losses_q[k + 1] += loss_q

                corrects[k + 1] = corrects[k + 1] + acc_q
                f1s[k + 1] = f1s[k + 1] + f1_q

        # end of all tasks
        # sum over all losses on query set across all tasks
        loss_q = losses_q[-1] / self.task_num
        if training == True:
            if torch.isnan(loss_q):
                pass
            else:
            # optimize theta parameters
                self.meta_optim.zero_grad()
                loss_q.backward(retain_graph=True)
                self.meta_optim.step()

        accs = np.array(corrects) / (self.task_num * querysz)
        f1_sc = np.array(f1s) / (self.task_num)

        return accs, f1_sc


class MLP(nn.Module):
    def __init__(self, hid):  # n_way = feature.shape[1]
        super(MLP, self).__init__()
        self.hidden = hid
        self.trans = nn.Linear(self.hidden, self.hidden)


    def forward(self, inputs): # inputs:[n_way, features.size(0)]
        params = self.trans(inputs)
        params = F.normalize(params, dim=-1)
        return params


class SCL(nn.Module):
    def __init__(self, in_fea, hid_fea, temperature=0.5):
        super(SCL, self).__init__()
        self.projector = nn.Sequential(
            nn.Linear(in_fea, hid_fea),
            nn.BatchNorm1d(hid_fea),
            nn.ReLU(inplace=True),
            nn.Linear(hid_fea, hid_fea))
        self.projector_2 = nn.Sequential(
            nn.Linear(in_fea, hid_fea),
            nn.BatchNorm1d(hid_fea),
            nn.ReLU(inplace=True),
            nn.Linear(hid_fea, hid_fea))
        self.tem = temperature

    def sup_contra(self, logits, mask, diagnal_mask=None):
        if diagnal_mask is None:
            diagnal_mask = 1 - diagnal_mask
            mask = mask * dignal_mask
            exp_logits = torch.exp(logits) * diagnal_mask
        else:
            exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = (-mean_log_prob_pos).mean()
        return loss

    def forward(self, fea, label, share=True):
        out = self.projector(fea)
        b = out.shape[0]
        device = out.device
        out = F.normalize(out, dim=1)
        label = label.contiguous().view(-1, 1)
        if share:
            mask = torch.eq(label, label.T).float().to(device)

        else:
            out = self.projector_2(doc_fea)
            out = F.normalize(out, dim=1)
            mask = torch.eq(label, label.T).float().to(device)
            
        diagnal_mask = torch.eye(b, b).to(device)
        scl_loss = self.sup_contra(out @ out.T / self.tem, mask, diagnal_mask)
        return scl_loss

