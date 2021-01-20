class DPGGAN_dblp2_Adam:
    def __init__(self):
        self.layer1_dim = 64
        self.layer2_dim = 64
        self.dec1_dim = 64
        self.dec2_dim = 64
        self.samp_num = 10
        self.delta = 1e-5
        self.eps_requirement = 10.0
        self.noise_sigma = 2.0
        self.batch_proc_size = 512
        self.grad_norm_max =  5
        self.C_decay = 0.99
        self.tol = 1.0


class DPGGAN_dblp2_ADADP:
    def __init__(self):
        self.layer1_dim = 32
        self.layer2_dim = 16
        self.dec1_dim = 16
        self.dec2_dim = 32
        self.samp_num = 10
        self.delta = 1e-5
        self.eps_requirement = 10.0
        self.noise_sigma = 5.0
        self.batch_proc_size = 512
        self.grad_norm_max =  5
        self.C_decay = 0.99
        self.tol = 1.0 # 'tolerance parameter'


class DPGVAE_dblp2_Adam:
    def __init__(self):
        self.layer1_dim = 64
        self.layer2_dim = 32
        self.dec1_dim = 32
        self.dec2_dim = 64
        self.samp_num = 10
        self.delta = 1e-5
        self.eps_requirement = 1.0
        self.noise_sigma = 2.0
        self.batch_proc_size = 512
        self.grad_norm_max = 5
        self.C_decay = 0.99
        self.tol = 1.0
        
        
class DPGVAE_dblp2_ADADP:
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
        

class DPGGAN_imdb_Adam:
    def __init__(self):
        self.layer1_dim = 64
        self.layer2_dim = 64
        self.dec1_dim = 64
        self.dec2_dim = 64
        self.samp_num = 10
        self.delta = 1e-5
        self.eps_requirement = 10.0
        self.noise_sigma = 2.0
        self.batch_proc_size = 512
        self.grad_norm_max = 5
        self.C_decay = 0.99


class DPGVAE_imdb_Adam:
    def __init__(self):
        self.layer1_dim = 32
        self.layer2_dim = 16
        self.dec1_dim = 16
        self.dec2_dim = 32
        self.samp_num = 10
        self.delta = 1e-5
        self.eps_requirement = 0.1
        self.noise_sigma = 2.0
        self.batch_proc_size = 512
        self.grad_norm_max = 5
        self.C_decay = 0.99


class DPGGAN_imdb_ADADP:
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


class DPGVAE_imdb_ADADP:
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


class GVAE_dblp2:
    def __init__(self):
        self.layer1_dim = 128
        self.layer2_dim = 64
        self.dec1_dim = 64
        self.dec2_dim = 128
        self.samp_num = 10


class GVAE_imdb:
    def __init__(self):
        self.layer1_dim = 128
        self.layer2_dim = 64
        self.dec1_dim = 64
        self.dec2_dim = 128
        self.samp_num = 10


class GGAN_dblp2:
    def __init__(self):
        self.layer1_dim = 128
        self.layer2_dim = 64
        self.dec1_dim = 64
        self.dec2_dim = 128
        self.samp_num = 10


class GGAN_imdb:
    def __init__(self):
        self.layer1_dim = 128
        self.layer2_dim = 64
        self.dec1_dim = 64
        self.dec2_dim = 128
        self.samp_num = 10



def load_config(model_name, dataset_str, optimizer):
    if model_name == 'GVAE' and 'dblp' in dataset_str:
        return GVAE_dblp2()
    elif model_name == 'GGAN' and 'dblp' in dataset_str:
        return GGAN_dblp2()
    elif model_name == 'DPGGAN' and 'dblp' in dataset_str  and optimizer == 'ADADP':
        return DPGGAN_dblp2_ADADP()
    elif model_name == 'DPGGAN' and 'dblp' in dataset_str and optimizer == 'Adam':
        return DPGGAN_dblp2_Adam()
    elif model_name == 'DPGVAE' and 'dblp' in dataset_str  and optimizer == 'ADADP':
        return DPGVAE_dblp2_ADADP()
    elif model_name == 'DPGVAE' and 'dblp' in dataset_str and optimizer == 'Adam':
        return DPGVAE_dblp2_Adam()
    elif model_name == 'GVAE' and 'IMDB_MULTI' in dataset_str:
        return GVAE_imdb()
    elif model_name == 'GGAN' and 'IMDB_MULTI' in dataset_str:
        return GGAN_imdb()
    elif model_name == 'DPGGAN' and 'IMDB_MULTI' in dataset_str and optimizer == 'ADADP':
        return DPGGAN_imdb_ADADP()
    elif model_name == 'DPGGAN' and 'IMDB_MULTI' in dataset_str and optimizer == 'Adam':
        return DPGGAN_imdb_Adam()
    elif model_name == 'DPGVAE' and 'IMDB_MULTI' in dataset_str and optimizer == 'ADADP':
        return DPGVAE_imdb_ADADP()
    elif model_name == 'DPGVAE' and 'IMDB_MULTI' in dataset_str and optimizer == 'Adam':
        return DPGVAE_imdb_Adam()
    else:
        print("Unknown config...")
        exit(1)