
require 'torch'
require 'nn'
require 'cutorch'

Categorical = {}
KernelNet = {}
function Categorical:new (category_count, time_slices)
  
  --
  --	input: tensor of size <time_slices> * <category_count>
  --	output: tensor of size <time_slices>
  --
  --	weights and bias of category_kernel_0 are shared across each time step, and learn
  -- 	a time independent importance weighting
  --

  o = nn.DepthConcat(1)
  o.category_kernel_0 =  nn.Linear(category_count, 1)
  o:add(category_kernel_0)

  for i=1,time_slices-1 do
  	local kernel_copy = o.category_kernel_0:clone('weight', 'bias')
  	o:add(kernel_copy)
  end
  
  return o
end

function KernelNet:new(lookback, cluster_count, act_count)
  
  lookback = lookback or 30
  cluster_count = cluster_count or 120
  act_count = act_count or 6

  self = nn.Sequential()
  
  kernel_matrix_init = torch.Tensor(lookback*2+1,3):fill(1):cuda()
  temporal_ratio = nn.ParallelTable()
  conv_layer_top = nn.SpatialConvolutionMM(1,1,lookback*2+1,3,1,1,lookback,0)
  conv_layer_top.weight = kernel_matrix_init:viewAs(conv_layer_top.weight)
  conv_layer_top.bias = torch.Tensor({0}):viewAs(conv_layer_top.bias)
  conv_layer_clone_bott = conv_layer_top:clone('weight','bias')	
  
  


  all_vars = nn.ParallelTable()	       -- input is {[61 x 1], [61 x 120], [61 x 6]}  
  all_vars:add(nn.Identity())		   -- blood glucose vals (pass directly to temporal step)
  cluster_kern = Categorical:new(cluster_count, lookback * 2 + 1)
  act_kern = Categorical:new(act_count, lookback * 2 + 1)
  all_vars:add(cluster_kern)   -- latlng clusters
  all_vars:add(act_kern)     -- activity types

  self.cluster_kern = cluster_kern.category_kernel_0.weight
  self.act_kern = act_kern.category_kernel_0.weight

  all_vars_holder = nn.Sequential()    -- input is {[61 x 1], [61 x 120], [61 x 6]}  
  all_vars_holder:add(all_vars)
  all_vars_holder:add(nn.JoinTable(2)) -- output is [61 x 3]

  all_impulse_holder = nn.Identity()   -- input / output is [61 x 3]


  consolidation_table = nn.ParallelTable()
  consolidation_table:add(all_vars_holder)
  consolidation_table:add(all_impulse_holder)

  self:add(consolidation_table)

  temporal_ratio:add(conv_layer_top)
  temporal_ratio:add(conv_layer_clone_bott)

  self:add(temporal_ratio) -- expects {Tensor(1,1,(lookback*2+1,3), Tensor(1,1,(lookback*2+1,3)}
  self:add(nn.CDivTable())
  return self:clone()
end

function test()
	local test_net = KernelNet:new(30, 120, 6)
	print(test_net)
	print(test_net.act_kern)
end
