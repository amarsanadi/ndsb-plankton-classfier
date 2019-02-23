
square_side = 60

batchSize = 128  -- don't change while running
batchMultiple = 2 	-- don't change while running
learningRate  = 0.0005
learningRateDecay = 0.0001
momentum  = 0.9
dampening = 0
nesterov = false
coefL1 = 1e-4          --           L1 penalty on the weights
coefL2 = 1e-5         --           L2 penalty on the weights
weightDecay = 0

--model_data.params[epoch] = {batchSize=batchSize,batchMultiple=batchMultiple,learningRate=learningRate,learningRateDecay=learningRateDecay,
--	momentum=momentum,nesterov=nesterov,coefL1=coefL1,coefL2=coefL2}


saveModel = true

optimization =  "LBFGS" --       optimization: SGD | CG | LBFGS  --update: only SGD available
maxIter=3

-- SGD Settings 
graph_save_path = '/home/ubuntu/Dropbox/Graphs/AWS_A'

dropoutEnabled = true

augment = false   -- [[true or false controls whether the data set is augmented with rotations and inversions]]
augment_degree_intervals = 0
lexi_version = "V6_3"


useCuda = true

testCNN = false
testFullModel = false
addConvLayer = false
if testCNN == true or testFullModel==true then
  data_set = 'small'
else
  data_set = 'big' --'big' | 'small'
end
