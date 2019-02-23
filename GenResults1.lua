--Generate submission

require 'torch'
require 'image'
require 'nn'
require 'nnx'
require 'optim'
require 'lfs'
require 'cunn' --switching to cunn
require 'cunnx' 
require 'cutorch'--model:evaluate()
require 'io'
require 'file_exists'
--require 'gfx.js'
testing = false

square_side = 67

test_images_path = "../test"
test_images_names_file = "../test_image_names.t7"

model_names  = {'NDSB_Conv_V6_Model_mod.net'}

--copy to a square box, rotate by degrees, scale, copy to dest
function lrs_image(src)
	width = src:size()[2]
	height = src:size()[3]

	hyp = math.sqrt(width^2+height^2)

	s = src:storage()
	for i=1,s:size() do s[i] = 1-s[i] end
	
	dst_img_tmp = torch.DoubleTensor(hyp,hyp):fill(0)

	x = math.ceil((hyp-width)/2)
	y = math.ceil((hyp-height)/2)
	dst_subzone = dst_img_tmp[{{x,x+width-1},{y,y+height-1}}]	
	dst_subzone:copy(src)
	dst_img_tmp= image.scale(dst_img_tmp,square_side)
	dst_img = torch.Tensor(square_side,square_side):fill(0)
	dst_img = image.scale(dst_img,dst_img_tmp,'bilinear')
	
	return dst_img
end

for model_num=1,#model_names do
	model_data = nil
	collectgarbage()
	model_data = torch.load(model_names[model_num])



	if testing == true then
		test_size = 10
	else
		test_size = 130400
	end

	dofile("classes.lua")
	if not file_exists(test_images_names_file) then
		dofile('GenTestImagesNamesFile.lua')
	end

	test_image_names = torch.load(test_images_names_file)

	results = {} 

	print(#test_image_names, ' images to be classified.')
	if model_data['model'] ==nil then
		model = model_data
	else
		model = model_data['model']
	end
	model:evaluate()
	model:cuda()
	cuda_input = torch.CudaTensor(1,square_side,square_side)
	for i=1,#test_image_names do
		xlua.progress(i, #test_image_names)
		--if i % 10 == 0 then collectgarbage() end

		image_name = test_image_names[i]
		img_raw = image.load(test_images_path..'/'..image_name)
		img_alt = lrs_image(img_raw)
		cuda_input[{}] = img_alt
		mean = cuda_input:mean()
		std = cuda_input:std()
		cuda_input = cuda_input:add(-mean)
		cuda_input = cuda_input:mul(1/std)
		
		--gfx.image({img_raw,input})
		output = model:forward(cuda_input)
		results[#results+1] = output:clone()
		if i == test_size then break end
	end
	torch.save(string.sub(model_names[model_num],1,-5)..'_RESULTS.t7',results)
end

