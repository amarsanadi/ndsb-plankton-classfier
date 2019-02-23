



mean = 0.034973040926706
std = 0.12545872733362

function get_next_training_set(set_type) 
  shuffle = torch.randperm(train_set_size) 

  if set_type == 'test' then 
    set_size = math.ceil(1/10*train_set_size) 
  else 
    set_size = train_set_size 
    set_type = 'big'
  end

  print('Generating *'..set_type..'* data set. Size: '..set_size)

  next_batch_images = torch.Tensor(set_size,1,square_side,square_side)
  next_batch_labels = torch.Tensor(set_size,1)
 
  for i=1,set_size do
      xlua.progress(i, set_size)

      img = train_set.data[shuffle[i]][1]
      local rads = math.random(math.ceil(2*math.pi*1000000))/1000000
      dst_img=image.rotate(img,rads,'bilinear')
      dst_img = dst_img:double()

      dst_label = train_set.label[shuffle[i]][1]
      
      border_width = math.random(10)
      step_x = math.random(border_width)-1
      if(math.random(2)==1) then step_x = -step_x end
      step_y = math.random(border_width)-1
      if(math.random(2)==1) then step_y = -step_y end

      next_batch_images[i][1] = image.scale(next_batch_images[i][1],dst_img[{{border_width+step_x,-(border_width-step_x)},{border_width+step_y, -(border_width-step_y)}}],'bilinear')
      next_batch_labels[i][1] = dst_label
  end

  mean = next_batch_images:mean()
  std = next_batch_images:std()
  print("Mean: ", mean)
  print("Std: ", std)

  next_batch_images:add(-mean)
  next_batch_images:div(std)
  return next_batch_images,next_batch_labels
end

