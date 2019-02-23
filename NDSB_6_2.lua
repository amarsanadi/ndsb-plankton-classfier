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

print('Running on device: ' .. cutorch.getDeviceProperties(cutorch.getDevice()).name)

dofile('00_settings.lua')
dofile('01_ndsb_data.lua')
dofile('02_ndsb_model.lua')

train_data_multi_batch_cuda = torch.CudaTensor(batchMultiple*batchSize,1,square_side,square_side)
train_labels_multi_batch_cuda =  torch.CudaTensor(batchMultiple*batchSize,1)

function train()
  
  shuffle_data()

  epoch = epoch or 1

  local time = sys.clock()
  model:training()
  print('** Training **')
  print('** Lexi '..lexi_version .. ' Training Epoch: '..epoch..' **')
 
  print('tbl_train_inputs size '..#tbl_train_inputs)

  for bigBatch = 1,#tbl_train_inputs, batchMultiple*batchSize do

    collectgarbage()
    big_batch_size = math.min(batchMultiple*batchSize,#tbl_train_inputs-bigBatch)
    train_data_multi_batch = train_data[{{bigBatch,bigBatch+big_batch_size-1}}]
    train_labels_multi_batch = train_labels[{{bigBatch,bigBatch+big_batch_size-1}}]

    --print('from '..bigBatch..' to '..bigBatch+big_batch_size-1)
    
    train_data_multi_batch_cuda[{{1,big_batch_size}}] = train_data_multi_batch
    train_labels_multi_batch_cuda[{{1,big_batch_size}}] = train_labels_multi_batch
    for miniBatch = 1,big_batch_size,batchSize do
      xlua.progress(bigBatch+miniBatch-1, #tbl_train_inputs)
      collectgarbage()
      --print('  training '..miniBatch..' to '..math.min(miniBatch+batchSize,big_batch_size)) 

      local feval = function(x)
        -- get new parameters
        if x ~= parameters then
          parameters:copy(x)
        end

        -- reset gradients
        gradParameters:zero()
        --gfx.image(train_data_multi_batch_cuda[{{miniBatch,math.min(miniBatch+batchSize,big_batch_size)}}])
        outputs = model:forward(train_data_multi_batch_cuda[{{miniBatch,math.min(miniBatch+batchSize,big_batch_size)}}])
        f = criterion:forward(outputs, train_labels_multi_batch_cuda[{{miniBatch,math.min(miniBatch+batchSize,big_batch_size)}}])

        df_do = criterion:backward(outputs, train_labels_multi_batch_cuda[{{miniBatch,math.min(miniBatch+batchSize,big_batch_size)}}])

        model:backward(train_data_multi_batch_cuda[{{miniBatch,math.min(miniBatch+batchSize,big_batch_size)}}], df_do)

        -- update confusion
        confusion:zero()
        for i=miniBatch,math.min(miniBatch+batchSize,big_batch_size) do
          confusion:add(outputs[i-miniBatch+1], train_labels_multi_batch_cuda[i][1])
        end

        if coefL1 ~= 0 or coefL2 ~= 0 then
          -- locals:
          local norm,sign= torch.norm,torch.sign

          -- Loss:
          f = f + coefL1 * norm(parameters,1)
          f = f + coefL2 * norm(parameters,2)^2/2

          -- Gradients:
          gradParameters:add( sign(parameters):mul(coefL1) + parameters:clone():mul(coefL2) )
        end
        if optimization == "SGD" then
        -- return f and df/dX
          return f,gradParameters
        else
          return f,gradParameters:double()
        end
      end

      -- optimize on current mini-batch


      if optimization=='SGD' then
        if nesterov==true then dampening=0 end
        config = config or {learningRate = learningRate, weightDecay = weightDecay, momentum = momentum,
                                learningRateDecay = learningRateDecay,nesterov = nesterov,dampening = dampening}

        optim.sgd(feval, parameters, config)
      else
          lbfgsState = lbfgsState or {
            learningRate = learningRate,
            maxIter = maxIter,
            lineSearch = lineSearch
          }
        optim.lbfgs(feval, parameters:double(), lbfgsState)
        collectgarbage()
      end

    end
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
  model_data.epoch = epoch

  if confusion.totalValid * 100 > maxTrainingMeanClassAccuracy then
    maxTrainingMeanClassAccuracy = confusion.totalValid * 100
    print('New Training Mean Class Accuracy Record!! '..(confusion.totalValid * 100)..' %')
    model_data.maxTrainingMeanClassAccuracy = maxTrainingMeanClassAccuracy
    print('Saving model to -'..lexi_version..'_MaxTraining_Model.net')
    torch.save(lexi_version..'_MaxTraining_Model.net', model_data)
  end




  --if #model_data.train_set_global_correct > 1 then
  print ('train_set_global_correct',model_data.train_set_global_correct)
  print ('train_set_average_row_correct',model_data.train_set_average_row_correct)
  print ('train_set_average_rowUcol_correct',model_data.train_set_average_rowUcol_correct)
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
  confusion:zero()
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

  if confusion.totalValid * 100 > maxTestingMeanClassAccuracy then
    maxTestingMeanClassAccuracy = confusion.totalValid * 100
    print('New Testing Mean Class Accuracy Record!! '..(confusion.totalValid * 100)..' %')
    model_data.maxTestingMeanClassAccuracy = maxTestingMeanClassAccuracy
    print('Saving model to - MaxTesting_Model.net')
    torch.save('MaxTesting_Model.net', model_data)
  end

  print ('test_set_global_correct',model_data.test_set_global_correct)
  print ('test_set_average_row_correct', model_data.test_set_average_row_correct)
  print ('test_set_average_rowUcol_correct', model_data.test_set_average_rowUcol_correct)
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
  dofile('00_settings.lua')
  print('batchSize: '..batchSize..'\nsaveModel: '..tostring(saveModel)..
    '\nlearningRate: '..learningRate..'\nlearningRateDecay: '..learningRateDecay..
    '\nmomentum: '..momentum..' Dampening: '..dampening..'\n L1 coef: '..coefL1..'\n L2 coef: '..coefL2..
    '\nNesterov: '..tostring(nesterov)..'\ngraph_save_path: '..graph_save_path)

  train()
  test()

  if saveModel == true then
      model_data.model = model
      print('<trainer> saving network to '..model_save_filename)
      torch.save(model_save_filename, model_data)
  end

--  savegraphs()

end

