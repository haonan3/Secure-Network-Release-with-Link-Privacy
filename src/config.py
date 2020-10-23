class DPGraphGAN_cora_Adam:
    def __init__(self):
        self.layer1_dim = 128
        self.layer2_dim = 32
        self.dec1_dim = 32
        self.dec2_dim = 64
        self.samp_num = 10
        self.delta = 1e-5
        self.eps_requirement = 1.0
        self.noise_sigma = 5.0
        self.batch_proc_size = 512
        self.grad_norm_max =  5
        self.C_decay = 0.99


class DPGraphGAN_cora_ADADP:
    def __init__(self):
        self.layer1_dim = 128
        self.layer2_dim = 32
        self.dec1_dim = 32
        self.dec2_dim = 64
        self.samp_num = 10
        self.delta = 1e-5
        self.eps_requirement = 1.0
        self.noise_sigma = 5.0
        self.batch_proc_size = 512
        self.grad_norm_max =  5
        self.C_decay = 0.99
        self.tol = 1.0 # 'tolerance parameter'


class DPGraphGAN_dblp2_Adam:
    def __init__(self):
        self.layer1_dim = 128
        self.layer2_dim = 128
        self.dec1_dim = 128
        self.dec2_dim = 128
        self.samp_num = 10
        self.delta = 1e-5
        self.eps_requirement = 1.0
        self.noise_sigma = 3.0
        self.batch_proc_size = 512
        self.grad_norm_max =  5
        self.C_decay = 0.99
        self.tol = 1.0


class DPGraphGAN_dblp2_ADADP:
    def __init__(self):
        self.layer1_dim = 32
        self.layer2_dim = 16
        self.dec1_dim = 16
        self.dec2_dim = 32
        self.samp_num = 10
        self.delta = 1e-5
        self.eps_requirement = 1.0
        self.noise_sigma = 5.0
        self.batch_proc_size = 512
        self.grad_norm_max =  5
        self.C_decay = 0.99
        self.tol = 1.0 # 'tolerance parameter'


class DPGraphVAE_dblp2_Adam:
    def __init__(self):
        self.layer1_dim = 64
        self.layer2_dim = 32
        self.dec1_dim = 32
        self.dec2_dim = 64
        self.samp_num = 10
        self.delta = 1e-5
        self.eps_requirement = 1.0
        self.noise_sigma = 4.0
        self.batch_proc_size = 512
        self.grad_norm_max = 5
        self.C_decay = 0.99
        self.tol = 1.0
        
        
class DPGraphVAE_dblp2_ADADP:
    def __init__(self):
        self.layer1_dim = 32
        self.layer2_dim = 16
        self.dec1_dim = 16
        self.dec2_dim = 32
        self.samp_num = 10
        self.delta = 1e-5
        self.eps_requirement = 1.0
        self.noise_sigma = 5.0
        self.batch_proc_size = 512
        self.grad_norm_max =  5
        self.C_decay = 0.99
        self.tol = 1.0 # 'tolerance parameter'
        

class DPGraphGAN_imdb_Adam:
    def __init__(self):
        self.layer1_dim = 64
        self.layer2_dim = 32
        self.dec1_dim = 32
        self.dec2_dim = 64
        self.samp_num = 10
        self.delta = 1e-5
        self.eps_requirement = 1.0
        self.noise_sigma = 5.0
        self.batch_proc_size = 512
        self.grad_norm_max = 5
        self.C_decay = 0.99


class DPGraphGAN_imdb_ADADP:
    def __init__(self):
        self.layer1_dim = 32
        self.layer2_dim = 16
        self.dec1_dim = 16
        self.dec2_dim = 32
        self.samp_num = 10
        self.delta = 1e-5
        self.eps_requirement = 1.0
        self.noise_sigma = 5.0
        self.batch_proc_size = 512
        self.grad_norm_max = 5
        self.C_decay = 0.99
        self.tol = 1.0  # 'tolerance parameter'
        
        
class DPGraphVAE_imdb_Adam:
    def __init__(self):
        self.layer1_dim = 32
        self.layer2_dim = 16
        self.dec1_dim = 16
        self.dec2_dim = 32
        self.samp_num = 10
        self.delta = 1e-5
        self.eps_requirement = 1.0
        self.noise_sigma = 5.0
        self.batch_proc_size = 512
        self.grad_norm_max = 5
        self.C_decay = 0.99
        

class DPGraphVAE_imdb_ADADP:
    def __init__(self):
        self.layer1_dim = 32
        self.layer2_dim = 16
        self.dec1_dim = 16
        self.dec2_dim = 32
        self.samp_num = 10
        self.delta = 1e-5
        self.eps_requirement = 1.0
        self.noise_sigma = 5.0
        self.batch_proc_size = 512
        self.grad_norm_max = 5
        self.C_decay = 0.99
        self.tol = 1.0  # 'tolerance parameter'
        

class GraphVAE_cora:
    def __init__(self):
        self.layer1_dim = 64
        self.layer2_dim = 32
        self.dec1_dim = 32
        self.dec2_dim = 64
        self.samp_num = 10


class GraphVAE_dblp2:
    def __init__(self):
        self.layer1_dim = 64
        self.layer2_dim = 32
        self.dec1_dim = 32
        self.dec2_dim = 64
        self.samp_num = 10


class GraphVAE_imdb:
    def __init__(self):
        self.layer1_dim = 256
        self.layer2_dim = 128
        self.dec1_dim = 128
        self.dec2_dim = 256
        self.samp_num = 10



def load_config(model_name, dataset_str, optimizer):
    if model_name == 'GraphVAE' and dataset_str == 'cora':
        return GraphVAE_cora()
    elif model_name == 'DPGraphGAN' and dataset_str == 'cora' and optimizer == 'ADADP':
        return DPGraphGAN_cora_ADADP()
    elif model_name == 'DPGraphGAN' and dataset_str == 'cora' and optimizer == 'Adam':
        return DPGraphGAN_cora_Adam()
    elif model_name == 'GraphVAE' and dataset_str in ['dblp2', 'new_dblp2', 'relabeled_dblp2']:
        return GraphVAE_dblp2()
    elif model_name == 'DPGraphGAN' and dataset_str in ['dblp2', 'new_dblp2', 'relabeled_dblp2']  and optimizer == 'ADADP':
        return DPGraphGAN_dblp2_ADADP()
    elif model_name == 'DPGraphGAN' and dataset_str in ['dblp2', 'new_dblp2', 'relabeled_dblp2'] and optimizer == 'Adam':
        return DPGraphGAN_dblp2_Adam()
    elif model_name == 'DPGraphVAE' and dataset_str in ['dblp2', 'new_dblp2', 'relabeled_dblp2']  and optimizer == 'ADADP':
        return DPGraphVAE_dblp2_ADADP()
    elif model_name == 'DPGraphVAE' and dataset_str in ['dblp2', 'new_dblp2', 'relabeled_dblp2'] and optimizer == 'Adam':
        return DPGraphVAE_dblp2_Adam()
    elif model_name == 'GraphVAE' and dataset_str in ['Resampled_IMDB_MULTI', 'new_IMDB_MULTI', 'IMDB_MULTI']:
        return GraphVAE_imdb()
    elif model_name == 'DPGraphGAN' and dataset_str in ['Resampled_IMDB_MULTI', 'new_IMDB_MULTI', 'IMDB_MULTI'] and optimizer == 'ADADP':
        return DPGraphGAN_imdb_ADADP()
    elif model_name == 'DPGraphGAN' and dataset_str in ['Resampled_IMDB_MULTI', 'new_IMDB_MULTI', 'IMDB_MULTI'] and optimizer == 'Adam':
        return DPGraphGAN_imdb_Adam()
    elif model_name == 'DPGraphVAE' and dataset_str in ['Resampled_IMDB_MULTI', 'new_IMDB_MULTI', 'IMDB_MULTI'] and optimizer == 'ADADP':
        return DPGraphVAE_imdb_ADADP()
    elif model_name == 'DPGraphVAE' and dataset_str in ['Resampled_IMDB_MULTI', 'new_IMDB_MULTI', 'IMDB_MULTI'] and optimizer == 'Adam':
        return DPGraphVAE_imdb_Adam()
    else:
        print("Unknown config...")
        exit(1)