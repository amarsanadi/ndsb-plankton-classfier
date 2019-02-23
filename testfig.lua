require 'gnuplot'



lexi_version='test'
graph_save_path=''
function savegraphs()
  --print ('** Saving training graph to: '..graph_save_path..'/lexi_'..lexi_version..'_training.png')
  
  local pngfigure = gnuplot.pngfigure(graph_save_path..'/lexi_'..lexi_version..'_training.png')
  gnuplot.plot({'Average Row Correct %',torch.Tensor(model_data.train_set_average_row_correct)},
    {'average rowUcol correct (VOC measure) %',torch.Tensor(model_data.train_set_average_rowUcol_correct)},
    {'Global Correct %',torch.Tensor(model_data.train_set_global_correct)})
  gnuplot.xlabel('Epoch')
  gnuplot.ylabel('%')
  gnuplot.title('Training Results')
  gnuplot.plotflush()
  gnuplot.close(pngfigure) 

  --print('** Saving test graph to: '..graph_save_path..'/lexi_'..lexi_version..'_testing.png')
  local pngfigure = gnuplot.pngfigure(graph_save_path..'/lexi_'..lexi_version..'_testing.png')
  gnuplot.plot({'Average Row Correct %',torch.Tensor(model_data.test_set_average_row_correct)},
    {'average rowUcol correct (VOC measure) %',torch.Tensor(model_data.test_set_average_rowUcol_correct)},
    {'Global Correct %',torch.Tensor(model_data.test_set_global_correct)})
  gnuplot.xlabel('Epoch')
  gnuplot.ylabel('%')
  gnuplot.title('Test Results')
  gnuplot.plotflush()
  gnuplot.close(pngfigure) 

end
model_data = {}
model_data.train_set_global_correct = {0,1,2,3,4,5,6}
model_data.train_set_average_row_correct = {0,1,2,3,4,5,6} 
model_data.train_set_average_rowUcol_correct = {0,1,2,3,4,5,6} 
model_data.test_set_global_correct = {0,1,2,3,4,5,6}
model_data.test_set_average_row_correct = {0,1,2,3,4,5,6}
model_data.test_set_average_rowUcol_correct = {0,1,2,3,4,5,6}


for i=1,100000 do

	savegraphs()
end