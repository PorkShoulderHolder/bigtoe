
require 'torch'
require 'math'
require 'nn'
require 'cutorch'
require 'kernelutil'

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

KernelNet = nn.Sequential()

function KernelNet:new (o, lookback, cluster_count, act_count, mask, buffer)
      local o = o or {}
      local lookback = lookback or 30
      local cluster_count = cluster_count or 120
      local act_count = act_count or 6
      local mask = mask or 2*lookback + 1
      local buffer = buffer or 0

      setmetatable(o, self)
      self.__index = self
      self.__tostring = o.__tostring
      self:init(lookback, cluster_count, act_count, mask)
      return o
end


function KernelNet:init(lookback, cluster_count, act_count, mask, buffer)
  
  self.lookback = lookback or 30
  self.cluster_count = cluster_count or 120
  self.act_count = act_count or 6
  self.buffer = buffer or 0

  self.mask = mask or 2*lookback + 1

  self.criterion = nn.MSECriterion():cuda()
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
  all_vars:add(cluster_kern)           -- latlng clusters
  all_vars:add(act_kern)               -- activity types

  self.cluster_kern = cluster_kern.category_kernel_0.weight
  self.act_kern = act_kern.category_kernel_0.weight

  all_vars_holder = nn.Sequential()    -- input is {[61 x 1], [61 x 120], [61 x 6]}  
  all_vars_holder:add(all_vars)
  all_vars_holder:add(nn.JoinTable(3)) -- output is [61 x 3]

  all_impulse_holder = nn.Identity()   -- input / output is [61 x 3]


  consolidation_table = nn.ParallelTable()
  consolidation_table:add(all_vars_holder)
  consolidation_table:add(all_impulse_holder)

  self:add(consolidation_table)

  temporal_ratio:add(conv_layer_top)
  temporal_ratio:add(conv_layer_clone_bott)

  self:add(temporal_ratio) -- expects {Tensor(1,1,(lookback*2+1,3), Tensor(1,1,(lookback*2+1,3)}
  self:add(nn.CDivTable())
end

function KernelNet:format( sample )

	--
	-- takes a sample of the form 31 x 3 and outputs 3 vectors
	-- 31 x 1
	-- 31 x 120
	-- 31 x 6
	--
	local act_hist = build_hist(sample[{{},3}], self.act_count)
	local loc_hist = build_hist(sample[{{},2}], self.cluster_count)
	local val_input = {sample[{{},1}]:clone(), loc_hist, act_hist}
	local impulses = sample:ne(0)

 	local out =  {val_input, impulses}
 	return out

end

function KernelNet:train( data )
	local shuffle_idxs = torch.randperm(data:size(1) - 2*self.lookback):add(self.lookback)

	for idx=1, data:size(1) do
		
		local i = shuffle_idxs[idx] 

		local sample = data[{{i - self.lookback, i + self.lookback}}]
		local input = self:format(sample)
		local target = input[1][self.mask]
	
		input[1][1][{{self.mask - self.buffer, self.mask}}]:fill(0)

		input[2][{{self.mask - self.buffer, self.mask},1}]:fill(0)
		
		local output = self:forward(input)

		local mseloss = self.criterion:forward(output, target)
		local gradient = self.criterion:backward(output,target)


	end
end

function KernelNet:testit()
	print("ok")
end

local args = {...} 

function test()
	local test_net = KernelNet:new()
	x = assert(loadfile('readjson.lua'))(args[1])
	print(x:size())
	test_net:train(x)
end

test()



