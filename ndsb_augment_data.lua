
tbl_train_inputs = {}


for i=1,train_set_size do
  if augment == true then
    for angle = 1,math.ceil(360/augment_degree_intervals)-1 do
      deg = angle*augment_degree_intervals
      tbl_train_inputs[#tbl_train_inputs+1] = {index = i,h=0,v=0,d=deg}
      tbl_train_inputs[#tbl_train_inputs+1] = {index = i,h=0,v=1,d=deg}
      tbl_train_inputs[#tbl_train_inputs+1] = {index = i,h=1,v=0,d=deg}
      tbl_train_inputs[#tbl_train_inputs+1] = {index = i,h=1,v=1,d=deg}
    end
  else 
    tbl_train_inputs[#tbl_train_inputs+1] = {index = i,h=0,v=0,d=0}
  end
end
print(#tbl_train_inputs)
