--train NDSB Model 
--Load the prepped training images and start training on the model, save the model after each epoch

require 'torch'
require 'image'
require 'nn'
require 'nnx'
require 'optim'
require 'cunn'
require 'cutorch'
require 'gnuplot'

require 'next_set'
require 'classes'
require 'file_exists'
require 'gfx.js'
testCNN = false
testFullModel = false
addConvLayer = false
if testCNN == true or testFullModel==true then
  data_set = 'small'
else
  data_set = 'big' --'big' | 'small'
end
dofile('settings.lua')
lexi_version = "V6" 

--model parameters

useCuda = true
if useCuda then
  require 'cunn' --switching to cunn
  require 'cutorch'

end
--cutorch.deviceReset()
print('Running on device: ' .. cutorch.getDeviceProperties(cutorch.getDevice()).name)
last_correct = 0
best_correct = 0

model_save_filename = 'NDSB_Conv_'..lexi_version..'_Model.net'

if file_exists(model_save_filename) == true then 
  loadsavedModel = true
else 
  loadsavedModel = false
end

t0=3

training_set_file_name='prepped_images_v6_'..data_set..'.t7'

print('Loading training data')

data = torch.load(training_set_file_name)
train_set = {}

train_set.data = data.training_images
train_set.label = data.training_labels
--gfx.image(train_set.data[{{1,20}}])
--mean=train_set.data:mean()
--std = train_set.data:std()
--train_set.data:add(-mean)
--train_set.data:div(std)


test_set_inputs  = data.testing_images
test_set_targets = data.testing_labels

mean = test_set_inputs:mean()
std = test_set_inputs:std()
  
test_set_inputs:add(-mean)
test_set_inputs:div(std)
  

train_set_size = train_set.data:size()[1]
test_set_size = test_set_inputs:size()[1]
print('Train set size '..train_set_size)

dofile('augment.lua')

--gfx.image(tbl_train_inputs[1].index)

--os.exit(0)
square_side = 65

noutputs = #classes
ninputs = square_side*square_side

--plotting
train_set_global_correct = {}
train_set_average_row_correct = {}
test_set_global_correct = {}
test_set_average_row_correct = {} 

if loadsavedModel==true then 
  print('Loading Saved Model from '..model_save_filename)
  model_data = torch.load(model_save_filename)
  
  model = model_data['model']

  train_set_global_correct = model_data['train_set_global_correct']
  train_set_average_row_correct = model_data['train_set_average_row_correct']
  train_set_average_rowUcol_correct = model_data['train_set_average_rowUcol_correct']

  test_set_global_correct = model_data['test_set_global_correct']
  test_set_average_row_correct = model_data['test_set_average_row_correct']
  test_set_average_rowUcol_correct = model_data['test_set_average_rowUcol_correct']
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
    --testCNN = true
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
  train_set_global_correct = {}
  train_set_average_row_correct = {} 
  train_set_average_rowUcol_correct = {}
  test_set_global_correct = {}
  test_set_average_row_correct = {}
  test_set_average_rowUcol_correct = {} 
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

  CNN:add(nn.SpatialConvolutionMM(256,384,5,5,1,1)) 
  CNN:add(nn.ReLU())

  CNN:add(nn.SpatialConvolutionMM(384,384,4,4,1,1)) 
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
  hidden_layer = next_layer*1.5
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
  print(model)

  print('************* testing model ***************')
  -- test_set_inputs, test_set_targets

  local temp_test_size = 128

  inputs = test_set_inputs[{{1,temp_test_size},{1},{},{}}]
  targets = test_set_targets[{{1,temp_test_size}}]

  gfx.image(inputs,{legend=''})
  model = model:cuda()
  inputs = inputs:cuda()
  targets = targets:cuda()

  model:training()
   --outputs = model:forward(inputs)
  confusion = optim.ConfusionMatrix(classes)

  parameters,gradParameters = model:getParameters()
  outputs = model:forward(inputs)
  print('Outputs size: ',outputs:size())
  print('************* model test complete *******************')
  if testFullModel == true then
    os.exit(0)
  end

  model = model:cuda()
  model_data = {}
  model_data.model = model
  model_data.lexi_version = lexi_version
  model_data.train_set_global_correct = {}
  model_data.train_set_average_row_correct = {} 
  model_data.train_set_average_rowUcol_correct = {} 
  model_data.test_set_global_correct = {}
  model_data.test_set_average_row_correct = {}
  model_data.test_set_average_rowUcol_correct = {}
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

if useCuda then
  
  model:cuda()
  criterion:cuda()
  --
end

--result = model:forward(testImage)

--print('*** Test image result  size: ', result:size())

-- retrieve parameters and gradients
parameters,gradParameters = model:getParameters()

print(model)

-- this matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

-- log results to files
trainLogger = optim.Logger('train.log')
testLogger = optim.Logger('test.log')
batchTrainCount = 0
--training_set_inputs, training_set_targets =  get_next_training_set()
--mean = training_set_inputs:mean()
--std = training_set_inputs:std()
--training_set_inputs:add(-mean)
--training_set_inputs:div(std)

--os.exit(1)
-- training function

function train()
  epoch = epoch or 1

  print('*** Lexi '..lexi_version .. ' Beginning Training. Getting Epoch number: '..epoch..' Training Data Set ***')
 

  train_set_size = #tbl_train_inputs

  shuffle = torch.randperm(#tbl_train_inputs)

  local time = sys.clock()
  model:training()
  print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. batchSize.. ']')

  print('*** Beginning Training ***')

  data = torch.Tensor(2500,1,65,65):cuda()
  for t = 1,train_set_size,batchSize do
    collectgarbage()
    -- disp progress
    xlua.progress(t, train_set_size)

    -- create mini batch
    this_batch_size = math.min(batchSize,train_set_size-t+1)
    --print('this_batch_size ', this_batch_size)
    inputs = torch.Tensor(this_batch_size,1,square_side,square_side) 
    targets = torch.Tensor(this_batch_size,1)
    --inputs = training_set_inputs[{{t,range_upper}}]
    --targets = training_set_targets[{{t,range_upper}}]
    for tbl_item = t,t+this_batch_size-1 do
      --print(tbl_item)
      entry = tbl_train_inputs[shuffle[tbl_item]]
      img = train_set.data[entry.index][1]:clone()
      if entry.h == 1 then
        img = image.hflip(img)
      end
      if entry.v==1 then
        img = image.vflip(img)
      end
      rads = entry.d / 2*math.pi
      if rads ~= 0 then img=image.rotate(img,rads,'bilinear') end
      label = train_set.label[entry.index][1]
      inputs[tbl_item-t+1] = img
      targets[tbl_item-t+1][1] = label  
    end
    --gfx.image(inputs[{{1,10}}])
    inputs = inputs:cuda()
    mean = inputs:mean()
    std = inputs:std()
    inputs = inputs:add(-mean)
    inputs = inputs:mul(1/std)

    targets = targets:cuda()

    --outputs = model:forward(inputs)

    -- create closure to evaluate f(X) and df/dX
    local feval = function(x)
      --collectgarbage()

      -- get new parameters
      if x ~= parameters then
        parameters:copy(x)
      end

      -- reset gradients
      gradParameters:zero()

      -- f is the average of all criterions
      local f = 0

      inputs = inputs:cuda()
      outputs = model:forward(inputs)
      f = criterion:forward(outputs, targets)

      df_do = criterion:backward(outputs, targets)

      model:backward(inputs, df_do)
      -- update confusion
      confusion:zero()
      for i=1,inputs:size()[1] do
       confusion:add(outputs[i], targets[i][1])
      end
       --confusion:add(outputs, targets[{{1,k},{1}}])

      if coefL1 ~= 0 or coefL2 ~= 0 then
        -- locals:
        local norm,sign= torch.norm,torch.sign

        -- Loss:
        f = f + coefL1 * norm(parameters,1)
        f = f + coefL2 * norm(parameters,2)^2/2

        -- Gradients:
        gradParameters:add( sign(parameters):mul(coefL1) + parameters:clone():mul(coefL2) )
      end

        -- return f and df/dX
        return f,gradParameters
    end

    -- optimize on current mini-batch
    config = config or {learningRate = learningRate, weightDecay = weightDecay, momentum = momentum,
      learningRateDecay = learningRateDecay}
    optim.sgd(feval, parameters, config)
  end


  -- time taken
  time = sys.clock() - time
  time = time / train_set_size
  print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')

  -- print confusion matrix
  print(confusion)

  print(' + average row correct: ' .. (confusion.averageUnionValid*100) .. '% \n')
  print(' + average rowUcol correct (VOC measure): ' .. (confusion.averageUnionValid*100) .. '% \n')
  print(' + global correct: ' .. (confusion.totalValid*100) .. '%')

  trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}
  
  model_data.train_set_average_row_correct[#model_data.train_set_average_row_correct+1] = confusion.averageValid*100
  model_data.train_set_global_correct[#model_data.train_set_global_correct+1] = confusion.totalValid*100
  model_data.train_set_average_rowUcol_correct[#model_data.train_set_average_rowUcol_correct+1] = confusion.averageUnionValid*100

  --if #model_data.train_set_global_correct > 1 then
  print ('train_set_global_correct',model_data.train_set_global_correct)
  print ('train_set_average_row_correct',model_data.train_set_average_row_correct)
  print ('train_set_average_rowUcol_correct',model_data.train_set_average_rowUcol_correct)

  gnuplot.epsfigure(graph_save_path..'/lexi_'..lexi_version..'_training.png')
  gnuplot.plot({'Average Row Correct %',torch.Tensor(model_data.train_set_average_row_correct)},
    {'average rowUcol correct (VOC measure) %',torch.Tensor(model_data.train_set_average_rowUcol_correct)},
    {'Global Correct %',torch.Tensor(model_data.train_set_global_correct)})
  gnuplot.xlabel('Epoch')
  gnuplot.ylabel('%')
  gnuplot.plotflush()
  
  confusion:zero()

  inputs = nil 
  outputs = nil 
  targets = nil 

  -- next epoch
  epoch = epoch + 1
end

-- test function
  function test()
    -- local vars
    test_set_size = test_set_inputs:size()[1]
    test_set_inputs=test_set_inputs:cuda()

    -- averaged param use?
    if average then
      cachedparams = parameters:clone()
      parameters:copy(average)
    end

    -- set model to evaluate mode (for modules that differ in training and testing, like Dropout)
    model:evaluate()
    local time = sys.clock()
    -- test over given dataset
    print('** Testing **')
    for i = 1,test_set_size do
      -- disp progress
      xlua.progress(i, test_set_size)

      -- test sample
      confusion:add(model:forward(test_set_inputs[i]), test_set_targets[i][1])
    end

    -- timing
    time = sys.clock() - time
    time = time / test_set_size
    print("<trainer> time to test 1 sample = " .. (time*1000) .. 'ms')

    -- print confusion matrix
    print(confusion)
    print(' + average row correct: ' .. (confusion.averageValid*100) .. '% \n')
    print(' + average rowUcol correct (VOC measure): ' .. (confusion.averageUnionValid*100) .. '% \n')
    print(' + global correct: ' .. (confusion.totalValid*100) .. '%')
    testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}

    model_data.test_set_average_row_correct[#model_data.test_set_average_row_correct+1] = confusion.averageValid*100
    model_data.test_set_global_correct[#model_data.test_set_global_correct+1] = confusion.totalValid*100
    model_data.test_set_average_rowUcol_correct[#model_data.test_set_average_rowUcol_correct+1] = confusion.averageUnionValid*100

    --if #model_data.test_set_global_correct > 1 then
    print ('test_set_global_correct',model_data.test_set_global_correct)
    print ('test_set_average_row_correct', model_data.test_set_average_row_correct)
    print ('test_set_average_rowUcol_correct', model_data.test_set_average_rowUcol_correct)
    gnuplot.epsfigure(graph_save_path..'/lexi_'..lexi_version..'_testing.png')
    gnuplot.plot({'Average Row Correct %',torch.Tensor(model_data.test_set_average_row_correct)},
      {'average rowUcol correct (VOC measure) %',torch.Tensor(model_data.test_set_average_rowUcol_correct)},
      {'Global Correct %',torch.Tensor(model_data.test_set_global_correct)})
    gnuplot.xlabel('Epoch')
    gnuplot.ylabel('%')
    gnuplot.plotflush()
    --end

    confusion:zero()

    -- averaged param use?
    if average then
      -- restore parameters
      parameters:copy(cachedparams)
    end
  end

----------------------------------------------------------------------
-- and train!
--

while true do
  dofile('settings.lua')
  print('batchSize: '..batchSize..' saveModel: '..tostring(saveModel)..' \nOptimization: '..optimization..' learningRate: '..learningRate..' learningRateDecay: '..learningRateDecay..' momentum: '..momentum)
  print('graph_save_path: '..graph_save_path)

  train()
  test()


  if saveModel == true then
      model_data.model = model
      print('<trainer> saving network to '..model_save_filename)
      torch.save(model_save_filename, model_data)
  end
end

