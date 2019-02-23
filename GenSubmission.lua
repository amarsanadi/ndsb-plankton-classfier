require 'cutorch'
require 'classes'
require 'file_exists'

result_sets = {}
testing = false
results_file_name = {'NDSB_Conv_V6_Model_mod_RESULTS.t7'}

num_result_sets = #results_file_name
for i=1,num_result_sets do
	result_sets[#result_sets+1] = torch.load(results_file_name[1])
end

test_images_names_file = "../test_image_names.t7"

submission_file_name = 'NDSB_Conv_V6_Model_mod_SUBMISSION.csv'

if file_exists(submission_file_name) then
	os.execute('rm '..submission_file_name)
end
test_image_names = torch.load(test_images_names_file)

header = 'image,'..table.concat(classes, ",") .. "\n"
if testing == false then 
	file = io.open(submission_file_name, "w")
	io.output(file)
	io.write(header)
else 
	print(header)	
end




for image_index = 1,#test_image_names do
	
	if num_result_sets > 1 then 

		result = torch.CudaTensor(121):fill(0)
	
		for result_sets_idx = 1,num_result_sets  do
			result = result + result_sets[result_sets_idx][image_index]
		end
		result = result:mul(1/num_result_sets)
	else
		result = result_sets[1][image_index]
	end
	output_line = {}
	for i=1,result:size()[1] do
		output_line[#output_line+1] =  string.format("%.8f", math.exp(result[i]))
		
	end
	line = test_image_names[image_index]..','..table.concat(output_line, ",")
	if testing == false then 
		io.write(line.."\n")
	else 
		print(line)
	end
	if testing == true and image_index > 10 then break end
end

if testing == false then file:close() end

