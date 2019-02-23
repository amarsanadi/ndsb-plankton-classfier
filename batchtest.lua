--batchtest.lua

require 'cutorch'

batchMultiple = 5
batchSize = 128
square_side = 65

dofile('ndsb_data.lua')

print('big batch max size '..batchMultiple*batchSize)

train_set_size = train_data:size()[1]

print('Train set size '..train_set_size)
print(#tbl_train_inputs..' entries were created')

train_data_multi_batch_cuda = torch.CudaTensor(batchMultiple*batchSize,1,65,65)
train_labels_multi_batch_cuda =  torch.CudaTensor(batchMultiple*batchSize,1)

for bigBatch = 1,train_set_size, batchMultiple*batchSize do
	
	big_batch_size = math.min(batchMultiple*batchSize,train_set_size-bigBatch)
	train_data_multi_batch = train_data[{{bigBatch,bigBatch+big_batch_size-1}}]
	train_labels_multi_batch = train_labels[{{bigBatch,bigBatch+big_batch_size-1}}]

	print('from '..bigBatch..' to '..bigBatch+big_batch_size-1)
	train_data_multi_batch_cuda[{{1,big_batch_size}}] = train_data_multi_batch
	train_labels_multi_batch_cuda[{{1,big_batch_size}}] = train_labels_multi_batch
	for miniBatch = 1,big_batch_size,batchSize do
		print('  processing '..miniBatch..' to '..math.min(miniBatch+batchSize,big_batch_size)) 


	end
end