
lexi_version = "V6"

testCNN = false
testFullModel = false

square_side = 60

batchSize = 128   -- don't change while running
batchMultiple =8 	-- don't change while running
learningRate  = 5e-4
learningRateDecay = 1e-7
momentum  = 0.9
dampening = 0
nesterov = false
coefL1 = 0 --1e-6         --           L1 penalty on the weights
coefL2 = 0 --1e-7         --           L2 penalty on the weights
weightDecay = 0
graph_save_path = '.'
model_save_filename = 'Conv3_3_2_hidden_Model_aws.net'

lineSearch = optim.lswolfe
maxIter = 3

saveModel = true

optimization =  "SGD" --       optimization: SGD | CG | LBFGS  --update: only SGD available

dropoutEnabled = true

augment = true   -- [[true or false controls whether the data set is augmented with rotations and inversions]]
augment_degree_intervals = 0

useCuda = true

addConvLayer = false
if testCNN == true or testFullModel==true then
  data_set = 'small'
else
  data_set = 'big' --'big' | 'small'
end
