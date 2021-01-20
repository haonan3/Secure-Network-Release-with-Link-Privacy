import os
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.abspath(os.path.join(dir_path, os.pardir))


class stat_logger:
    def __init__(self, args, current_time):
        self.log_file_name = '{}_{}_{}.txt'.format(current_time, args.model_name, args.dataset_str)
        self.log_save_folder = root_path + '/log/txt_log/'
        self.save_path = self.log_save_folder + self.log_file_name


    def check_folder(self):
        pass

    def write(self, log_info):
        with open(self.save_path, 'a') as log_file:
            log_file.write(log_info + '\n')


    def form_generated_stat_log(self, epoch, property_cache):
        stat_vec_cache = []
        assert len(property_cache) > 1
        for property in property_cache:
            _, stat_vec = self.dict_to_vec(property)
            stat_vec_cache.append(stat_vec)
        stat_vec_mean = np.array(stat_vec_cache).mean(axis=0)
        stat_vec_str = ["%.3f" % number for number in stat_vec_mean]
        stat_vec_log = ' '.join(stat_vec_str)
        log = 'Epoch@{}: '.format(epoch) + stat_vec_log
        return log


    def from_dp_log(self, model):
        counter = model.dp_counter

    def form_original_stat_log(self, property):
        stat_name, stat_vec = self.dict_to_vec(property)
        stat_vec_str = ["%.3f" % number for number in stat_vec]
        stat_name_log = ' '.join(stat_name)
        stat_vec_log = ' '.join(stat_vec_str)
        return stat_name_log + '\n' + 'original_graph: ' + stat_vec_log


    def form_args_log_content(self, args, model_args):
        args_info_str = str(args).split('Namespace')[1].split('(')[1].split(')')[0]
        model_args_info_str = str(model_args.__dict__).split('{')[1].split('}')[0]
        return 'Args: {}.\nModel_Args: {}.\n'.format(args_info_str, model_args_info_str)


    def dict_to_vec(self, stat_dict):
        stat_name = list(stat_dict.keys())
        stat_vec = np.array(list(stat_dict.values()))
        return stat_name, stat_vec