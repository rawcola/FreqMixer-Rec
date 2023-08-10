from recbole.quick_start import run_recbole
parameter_dict = {
   'neg_sampling': None,
}
config_file_list = ['freqmixer.yaml']
run_recbole(model='FreqMixer', dataset='office', config_file_list=config_file_list, config_dict=parameter_dict)


