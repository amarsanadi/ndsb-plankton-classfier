require 'lfs'
require 'image'
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods

require 'classes.lua'
require 'lrs_image2.lua'

genSmallSet = false
smallSetSize = 1500
train_set = {}
counter = 0
fpath = '../train/'
square_side = 60

if genSmallSet == true then 
	file_ext = 'small'
else
	file_ext = 'big'
end

save_file_name = 'prepped_images_v3_'..file_ext..'.t7'
phyla_count = {}
phyla_start_index = {}
phyla_end_index = {}

for phyla_name in lfs.dir(fpath) do
	if phyla_name ~= '.' and phyla_name ~= '..' then
		
		pcounter = 0
		phyla_start_index[phyla_name] = counter+1
		for fnam in lfs.dir(fpath..phyla_name) do
			if fnam ~= '.' and fnam ~= '..' then
				counter = counter + 1
				pcounter = pcounter + 1
				im = {}
				im.class  = revClasses[phyla_name]
				image_path  = fpath..phyla_name..'/'..fnam
				img = image.load(image_path)
				im.image = img:float()
				im.name = fnam
				train_set[counter] = im
				phyla_end_index[phyla_name] = counter
			end
			if genSmallSet==true and counter == smallSetSize then break end
		end
		print(phyla_name..':'..pcounter..' images')
		phyla_count[phyla_name] = pcounter
	end
	if genSmallSet==true and counter == smallSetSize then break end
end

print(counter, ' files found')

--find the average number of images per phyla
avg = math.ceil(30336 / 121)

-- for each of the phyla
--	if the number of images is < the avg
--     randomly select images and duplicate with random rotations and horiz, vert transforms 
--[[
for k,v in pairs(phyla_count) do
	if v < avg then 
		print('less than average', k,v) 
		start_index =  phyla_start_index[k]
		end_index = phyla_end_index[k]
		num_images_to_generate = avg - v
		for i = 1,num_images_to_generate do
			--randomly pick an image from start_index to end_index
			index = start_index+torch.random(v)-1
			im = train_set[index]
			newim = {}
			newim.class = im.class

			newim.image = im.image:clone()
			if torch.random(2) == 1 then
				newim.image = image.hflip(newim.image)
			end
			if torch.random(2) == 1 then
				newim.image = image.vflip(newim.image)
			end
			newim.name = im.name

			train_set[#train_set+1] = im 
		end
	end
end
--]]

--os.exit(1)

shuffle = torch.randperm(#train_set) 

testing_data_size = math.ceil(10/100 * #train_set)
training_data_size = #train_set - testing_data_size

training_images_set = torch.Tensor(training_data_size,1,square_side,square_side)
training_labels_set = torch.Tensor(training_data_size,1)

testing_images_set = torch.Tensor(testing_data_size,1,square_side,square_side)
testing_labels_set = torch.Tensor(testing_data_size,1)

-- pick up each image and label
-- transform the image - put it on a square of side = hyp of the image
-- 
print('Generating testing data')
for i=1,testing_data_size do
	xlua.progress(i,testing_data_size)
	testing_images_set[i] = lrs_image2(train_set[shuffle[i]].image,square_side)
	testing_labels_set[i] = train_set[shuffle[i]].class
end

print('Generating training data')
for i=1,training_data_size do
	xlua.progress(i,training_data_size)
	training_images_set[i][1] = lrs_image2(train_set[shuffle[i+testing_data_size]].image,square_side)
	training_labels_set[i] = train_set[shuffle[i+testing_data_size]].class
end

data = {}
data['training_images'] = training_images_set
data['training_labels'] = training_labels_set
data['testing_images'] = testing_images_set
data['testing_labels'] = testing_labels_set

torch.save(save_file_name,data)


