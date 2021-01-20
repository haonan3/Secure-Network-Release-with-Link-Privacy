import torch
from collections import OrderedDict
from torch.autograd import Variable
from src.DPGGAN import gaussian_moments as gm, px_expander

'''
Update privacy budget

priv_pars: the privacy dictionary
'''
def update_privacy_pars(dp_counter):
    verify = False
    max_lmbd = 32
    lmbds = range(1, max_lmbd + 1)
    log_moments = []
    for lmbd in lmbds:
        log_moment = 0
        '''
        print('Here q = ' + str(priv_pars['q']))
        print('Here sigma = ' + str(priv_pars['sigma']))
        print('Here T = ' + str(priv_pars['T']))
        '''
        log_moment += gm.compute_log_moment(dp_counter.q, dp_counter.sigma, dp_counter.T, lmbd, verify=verify)
        log_moments.append((lmbd, log_moment))
    dp_counter.eps, _ = gm.get_privacy_spent(log_moments, target_delta=dp_counter.delta)
    return dp_counter


'''
create container for accumulated gradient

:return is the gradient container
'''
def create_cum_grads(model):
    cum_grads = OrderedDict()
    for i, p in enumerate(model.parameters()):
        if p.requires_grad:
            cum_grads[str(i)] = Variable(torch.zeros(p.shape[1:]), requires_grad=False)
    return cum_grads



def update_privacy_account(model_args, model):
    stop_signal = False
    if 'dp_counter' in set(model.__dict__.keys()):
        model.dp_counter.T += 1
        update_privacy_pars(model.dp_counter)
        model_args.grad_norm_max *= model_args.C_decay
        if model.dp_counter.eps > model_args.eps_requirement:
            model.dp_counter.should_stop = True
            stop_signal = model.dp_counter.should_stop
    return stop_signal



def perturb_grad(model_args, model):
    # For DP model: accumulate grads in the container, cum_grads; add noise on sum of grads
        px_expander.acc_scaled_grads(model=model, C=model_args.grad_norm_max, cum_grads=model.cum_grads)

        # because we don't use lot-batch structure, so just add noise after acc_grads
        px_expander.add_noise_with_cum_grads(model=model, C=model_args.grad_norm_max,
                                             sigma=model_args.noise_sigma, cum_grads=model.cum_grads,
                                             samp_num=model_args.samp_num)