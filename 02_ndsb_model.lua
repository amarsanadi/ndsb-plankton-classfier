
require 'cutorch'
require 'file_exists'
require 'classes'
require 'nn'
require 'nnx'
require 'cunn'

if train_data==nil then dofile('01_ndsb_data.lua') end

if lexi_version == nil then lexi_version = 'testing' end

--model_save_filename = 'NDSB_Conv_'..lexi_version..'_Model.net'

if file_exists(model_save_filename) == true then 
  loadsavedModel = true
else 
  loadsavedModel = false
end

t0=3

train_set_global_correct = {}
train_set_average_row_correct = {}
test_set_global_correct = {}
test_set_average_row_correct = {} 

if loadsavedModel==true then 
  print('Loading Saved Model from '..model_save_filename)
  model_data = torch.load(model_save_filename)
  
  model = model_data['model']
  print(model)
  train_set_global_correct = model_data['train_set_global_correct']
  train_set_average_row_correct = model_data['train_set_average_row_correct']
  train_set_average_rowUcol_correct = model_data['train_set_average_rowUcol_correct']

  test_set_global_correct = model_data['test_set_global_correct']
  test_set_average_row_correct = model_data['test_set_average_row_correct']
  test_set_average_rowUcol_correct = model_data['test_set_average_rowUcol_correct']
  maxTestingMeanClassAccuracy = model_data['maxTestingMeanClassAccuracy']
  if maxTestingMeanClassAccuracy == nil then maxTestingMeanClassAccuracy = 0 end
  maxTrainingMeanClassAccuracy = model_data['maxTrainingMeanClassAccuracy']
  if maxTrainingMeanClassAccuracy == nil then maxTrainingMeanClassAccuracy = 0 end

  epoch = #train_set_global_correct+1
  

  if addConvLayer == true then
    inputs = test_set_inputs[{{1,temp_test_size},{1},{},{}}]
    targets = test_set_targets[{{1,temp_test_size}}]

    CNN=model.modules[1]
    print(CNN)
    --classified = model.modules[2]:clone()

    CNN_out = CNN:forward(inputs[1]:cuda())
    print(CNN_out:size())

    CNN:add(nn.SpatialConvolutionMM(256,256,5,5,1,1,2)) 
    CNN:add(nn.ReLU())
    CNN:cuda()
    
    print(CNN)
    
    CNN_out = CNN:forward(inputs[1]:cuda())
    print(CNN_out:size())

    --next_layer = CNN_out:size()[2]*CNN_out:size()[3]*CNN_out:size()[4]
    --print(next_layer)

    if testCNN==true then
      os.exit(1)
    end

  end
else
  --train_set_global_correct = {0}
  --train_set_average_row_correct = {0} 
  --train_set_average_rowUcol_correct = {0}
  --test_set_global_correct = {0}
  --test_set_average_row_correct = {0}
  --test_set_average_rowUcol_correct = {0} 
  epoch=1
  print('Building model')

  CNN = nn.Sequential()

  CNN:add(nn.SpatialConvolutionMM(1,128,8,8,1,1)) 
  CNN:add(nn.ReLU())
  CNN:add(nn.SpatialMaxPooling(2,2,2,2))

  CNN:add(nn.SpatialConvolutionMM(128,256,6,6,1,1)) 
  CNN:add(nn.ReLU())

  CNN:add(nn.SpatialConvolutionMM(256,256,5,5,1,1)) 
  CNN:add(nn.ReLU())
  CNN:add(nn.SpatialMaxPooling(2,2,2,2))

  CNN:add(nn.SpatialConvolutionMM(256,256,5,5,1,1)) 
  CNN:add(nn.ReLU())

  CNN:add(nn.SpatialConvolutionMM(256,256,4,4,1,1)) 
  CNN:add(nn.ReLU())
  CNN:add(nn.SpatialMaxPooling(2,2,1,1))
--[[
  

  CNN:add(nn.SpatialConvolutionMM(256,384,4,4,1,1,2)) 
  CNN:add(nn.ReLU())
  CNN:add(nn.SpatialMaxPooling(3,3,2,2))

  CNN:add(nn.SpatialConvolutionMM(384,512,3,3,1,1,1)) 
  CNN:add(nn.ReLU())
  
  CNN:add(nn.SpatialConvolutionMM(512,512,3,3,1,1,1)) 
  CNN:add(nn.ReLU())
  --CNN:add(nn.SpatialMaxPooling(1,1,1,1)) 

  CNN:add(nn.SpatialConvolutionMM(512,512,3,3,1,1,1)) 
  CNN:add(nn.ReLU())  
--]]

  print(CNN)
  
  inputs = test_set_inputs[{{1,temp_test_size},{1},{},{}}]
  targets = test_set_targets[{{1,temp_test_size}}]
  
  CNN_out = CNN:forward(inputs[{{1},{1},{},{}}])
  print(CNN_out:size())

  next_layer = CNN_out:size()[2]*CNN_out:size()[3]*CNN_out:size()[4]
  print(next_layer)

  if testCNN==true then
    os.exit(1)
  end
  hidden_layer = next_layer*3
  classifier = nn.Sequential()

  classifier:add(nn.Reshape(next_layer))
-- classifier:add(nn.Dropout(0.5))
  classifier:add(nn.Linear(next_layer, hidden_layer))
  classifier:add(nn.ReLU())
  -- classifier:add(nn.Dropout(0.5))
  classifier:add(nn.Linear(hidden_layer, #classes))

  classifier:add(nn.LogSoftMax())

  for _,layer in ipairs(CNN.modules) do
    if layer.bias then
      layer.bias:fill(.2)
      if i == #CNN.modules-1 then
        layer.bias:zero()
      end
    end
  end

  model = nn.Sequential()
  model:add(CNN)
  model:add(classifier)


  model =  model:cuda()

  print(model)

  print('************* testing model ***************')
  -- test_set_inputs, test_set_targets

  local temp_test_size = 128

  inputs = test_set_inputs[{{1,temp_test_size},{1},{},{}}]
  targets = test_set_targets[{{1,temp_test_size}}]

  gfx.image(inputs,{legend=''})
  model = model:cuda()
  --inputs = inputs:cuda()
  --targets = targets:cuda()

  model:training()
   --outputs = model:forward(inputs)
  confusion = optim.ConfusionMatrix(classes)

  parameters,gradParameters = model:getParameters()
  --outputs = model:forward(inputs)
  print('Outputs size: ',outputs:size())
  print('************* model test complete *******************')
  --if testFullModel == true then
    --os.exit(0)
  --end

  model = model:cuda()
  model_data = {}
  model_data.model = model
  model_data.lexi_version = lexi_version
  model_data.train_set_global_correct = {0}
  model_data.train_set_average_row_correct = {0} 
  model_data.train_set_average_rowUcol_correct = {0} 
  model_data.test_set_global_correct = {0}
  model_data.test_set_average_row_correct = {0}
  model_data.test_set_average_rowUcol_correct = {0}

  model_data.epoch = 1
  model_data.params = {}
  model_data.params[model_data.epoch] = {batchSize=batchSize,batchMultiple=batchMultiple,learningRate=learningRate,
                                            learningRateDecay=learningRateDecay,momentum=momentum,nesterov=nesterov,
                                            coefL1=coefL1,coefL2=coefL2}


  if saveModel == true then
    torch.save(model_save_filename, model_data)
  else
    print('Not saving the model. ')
  end
end

----------------------------------------------------------------------
-- loss function: negative log-likelihood
--

criterion = nn.ClassNLLCriterion()
criterion = criterion:cuda()

parameters,gradParameters = model:getParameters()

print(model)

-- this matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

-- log results to files
trainLogger = optim.Logger(graph_save_path..'/lexi_'..lexi_version..'_train.txt')
testLogger = optim.Logger(graph_save_path..'/lexi_'..lexi_version..'_test.txt')
batchTrainCount = 0

function savegraphs()
  print ('** Saving training graph to: '..graph_save_path..'/lexi_'..lexi_version..'_training.eps')
  
  local epsfigure = gnuplot.epsfigure(graph_save_path..'/lexi_'..lexi_version..'_training.eps')
  gnuplot.plot({'Average Row Correct %',torch.Tensor(model_data.train_set_average_row_correct)},
    {'average rowUcol correct (VOC measure) %',torch.Tensor(model_data.train_set_average_rowUcol_correct)},
    {'Global Correct %',torch.Tensor(model_data.train_set_global_correct)})
  gnuplot.xlabel('Epoch')
  gnuplot.ylabel('%')
  gnuplot.title('Training Results')
  gnuplot.plotflush()
  gnuplot.close(epsfigure) 

  print('** Saving test graph to: '..graph_save_path..'/lexi_'..lexi_version..'_testing.eps')
  local epsfigure = gnuplot.epsfigure(graph_save_path..'/lexi_'..lexi_version..'_testing.eps')
  gnuplot.plot({'Average Row Correct %',torch.Tensor(model_data.test_set_average_row_correct)},
    {'average rowUcol correct (VOC measure) %',torch.Tensor(model_data.test_set_average_rowUcol_correct)},
    {'Global Correct %',torch.Tensor(model_data.test_set_global_correct)})
  gnuplot.xlabel('Epoch')
  gnuplot.ylabel('%')
  gnuplot.title('Test Results')
  gnuplot.plotflush()
  gnuplot.close(epsfigure) 

end

savegraphs()
