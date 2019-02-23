require 'cutorch'
require 'cunn'

model_name = 'NDSB_Conv_3_2_Model.net'

temp_data = torch.load(model_name)
model_data = {}
if temp_data['model'] == nil then
	model_data['model'] = temp_data
    model_data.lexi_version = lexi_version
    model_data.train_set_global_correct = {0}
    model_data.train_set_average_row_correct = {0} 
    model_data.train_set_average_rowUcol_correct = {0} 
  	model_data.test_set_global_correct = {0}
  	model_data.test_set_average_row_correct = {0}
  	model_data.test_set_average_rowUcol_correct = {0}
  	model_data.max_mean_class_accuracy_test = 0
  	model_data.max_mean_class_accuracy_train = 0
  	torch.save('NDSB_Conv_3_2_mod_Model.net',model_data)
 end

