
--lossless rotate and scale
--copy to a square box, rotate by degrees, scale, copy to dest
function lrs_image2(src,side)
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
	--dst_img_tmp= image.scale(dst_img_tmp,side)
	dst_img = torch.Tensor(side,side):fill(0)
	dst_img = image.scale(dst_img,dst_img_tmp,'bilinear')
	
	return dst_img
end