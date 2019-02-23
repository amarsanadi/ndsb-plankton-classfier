require 'image'
--require 'gfx.js'

if data_set ==nil then 
	data_set = 'small' 
	test_mode = true
end
if augment == nil then augment = true end
if square_side==nil then square_side = 60 end

training_set_file_name='prepped_images_v3_'..data_set..'.t7'

print('Loading training data')

data = torch.load(training_set_file_name)
train_set = {}

train_set.data = data.training_images

--gfx.image(train_set.data[{{1,10}}])
train_set.label = data.training_labels
--gfx.image(train_set.data[{{1,20}}])

mean=train_set.data:mean()
std = train_set.data:std()
train_set.data:add(-mean)
train_set.data:div(std)

test_set_inputs  = data.testing_images
test_set_targets = data.testing_labels

mean = test_set_inputs:mean()
std = test_set_inputs:std()
  
test_set_inputs:add(-mean)
test_set_inputs:div(std)
  

train_set_size = train_set.data:size()[1]
test_set_size = test_set_inputs:size()[1]
print('Train set size '..train_set_size)

tbl_train_inputs = {}

for i=1,train_set_size do
  if augment == true then
  	if augment_degree_intervals ~= 0 then
    	for angle = 1,math.ceil(360/augment_degree_intervals)-1 do
    		deg = angle*augment_degree_intervals
		    tbl_train_inputs[#tbl_train_inputs+1] = {index = i,h=0,v=0,d=deg}
   			
    	end
    		tbl_train_inputs[#tbl_train_inputs+1] = {index = i,h=1,v=1,d=0}
		    tbl_train_inputs[#tbl_train_inputs+1] = {index = i,h=0,v=1,d=0}
   			tbl_train_inputs[#tbl_train_inputs+1] = {index = i,h=1,v=0,d=0}

    else
    	deg=0
    	tbl_train_inputs[#tbl_train_inputs+1] = {index = i,h=0,v=0,d=0}
    	tbl_train_inputs[#tbl_train_inputs+1] = {index = i,h=0,v=1,d=0}
    	tbl_train_inputs[#tbl_train_inputs+1] = {index = i,h=1,v=0,d=torch.random(360)}
    	tbl_train_inputs[#tbl_train_inputs+1] = {index = i,h=1,v=1,d=torch.random(360)}
    	
    end
  else 
    tbl_train_inputs[#tbl_train_inputs+1] = {index = i,h=0,v=0,d=0}
  end
end
print('augmented data size: ',#tbl_train_inputs)

train_data = torch.Tensor(#tbl_train_inputs,1,square_side,square_side)
train_labels = torch.Tensor(#tbl_train_inputs,1)

function shuffle_data()
	shuffle = torch.randperm(#tbl_train_inputs)
	for tbl_item = 1,#tbl_train_inputs do

		entry = tbl_train_inputs[shuffle[tbl_item]]
		img = train_set.data[entry.index][1]:clone()

		if entry.h == 1 then
			img = image.hflip(img)
		end
		if entry.v==1 then
			img = image.vflip(img)
		end
		if entry.d ~=0 then
			rads = entry.d / 2*math.pi
			img=image.rotate(img,rads,'bilinear') 
		end
		label = train_set.label[entry.index][1]
		if img:size()[1] ~= square_side then
			--print(img:size())
			--print(train_data[{{tbl_item},{1}}]:size())
			img_tmp = torch.Tensor(square_side,square_side)
			img_tmp=image.scale(img_tmp,img,'bilinear')
			train_data[{{tbl_item},{1}}] = img_tmp
		else
			train_data[{{tbl_item},{1}}]  = img
		end
		train_labels[tbl_item][1] = train_set.label[entry.index][1]

	end
	train_labels = train_labels:float()
end

if test_mode == true then
	shuffle_data()
	require 'gfx.js'
	gfx.image(train_data[{{1,50}}])
end

