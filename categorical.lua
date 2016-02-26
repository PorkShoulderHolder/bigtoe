
require 'torch'
require 'math'
require 'nn'
require 'cunn'
require 'cutorch'
require 'kernelutil'
require 'CDivTable_rebust'
require 'gnuplot'
require 'debug'
require 'os'


local Categorical, parent = torch.class('nn.Categorical', 'nn.DepthConcat')

KernelNet = {}
function Categorical:__init (category_count, time_slices)
  
  --
  --	input: tensor of size <time_slices> * <category_count>
  --	output: tensor of size <time_slices>
  --
  --	weights and bias of category_kernel_0 are shared across each time step, and learn
  -- 	a time independent importance weighting
  --
  parent.__init(self,3)

 -- o = nn.DepthConcat(1)
  self.category_kernel_0 =  nn.Linear(category_count, 1)
  self:add(category_kernel_0)

  for i=1,time_slices do
  	local kernel_copy = self.category_kernel_0:clone('weight', 'bias')
  	self:add(kernel_copy)
  end
end

function Categorical:updateOutput(input)
	print(self)
	print(self.output)
	print("asdas")
	return self.output
end

KernelNet = nn.Sequential()

function KernelNet:new (o, lookback, cluster_count, act_count, mask, buffer, maxgrad)
      local o = o or {}
      setmetatable(o, self)
      self.__index = self
      self.__tostring = o.__tostring
      self:init(lookback, cluster_count, act_count, mask)
      return o
end


function KernelNet:init(lookback, cluster_count, act_count, mask, buffer)
  
  self:setupHyperParams(lookback,cluster_count,act_count,mask,buffer)
  self.criterion = nn.MSECriterion():cuda()
  kernel_matrix_init = torch.Tensor((self.lookback*2+1) * 3):fill(1)
  kernel_matrix_init[{{62,122}}]:fill(1)
  kernel_matrix_init[{{122,183}}]:fill(1)

  temporal_ratio = nn.ParallelTable()
  conv_layer_top = nn.Linear((self.lookback*2+1) * 3,1)
  conv_layer_top.weight = kernel_matrix_init:clone():viewAs(conv_layer_top.weight)
  conv_layer_top.bias = torch.Tensor({0}):viewAs(conv_layer_top.bias)
  conv_layer_clone_bott = conv_layer_top:clone('weight','bias')	
   
  all_vars = nn.ParallelTable()	       -- input is {[61 x 1], [61 x 120], [61 x 6]}  
  all_vars:add(nn.Identity())		   -- blood glucose vals (pass directly to temporal step)
  cluster_kern = nn.Linear(self.cluster_count, 1)
  act_kern = nn.Linear(self.act_count, 1)

  act_kern.weight = act_kern.weight:fill(1)
  cluster_kern.weight = cluster_kern.weight:fill(1	)

  all_vars:add(cluster_kern)           -- latlng clusters
  all_vars:add(act_kern)               -- activity types

  self.cluster_kern = cluster_kern.weight
  self.act_kern = act_kern.weight

  all_vars_holder = nn.Sequential()    -- input is {[61 x 1], [61 x 120], [61 x 6]}  
  all_vars_holder:add(all_vars)
  all_vars_holder:add(nn.JoinTable(1)) -- output is [61 x 3]
  all_vars_holder:add(nn.View(1,(2*self.lookback + 1) * 3))
  
  all_impulse_holder = nn.Sequential()
  all_impulse_holder:add(nn.View(1,(2*self.lookback + 1) * 3))  -- input / output is [61 x 3]

  zero_out = nn.Sequential()
  consolidation_table = nn.ParallelTable()
  consolidation_table:add(all_vars_holder)
  consolidation_table:add(all_impulse_holder)

  zero_out:add(consolidation_table)
  zero_out:add(nn.CMulTable())

  top_bottom = nn.ParallelTable()
  top_bottom:add(zero_out)
  top_bottom:add(all_impulse_holder)

  self:add(top_bottom)

  temporal_ratio:add(conv_layer_top)
  temporal_ratio:add(conv_layer_clone_bott)

  self:add(temporal_ratio) -- expects {Tensor(1,1,(lookback*2+1,3), Tensor(1,1,(lookback*2+1,3)}
  self:add(CDivTable_robust())
  conv_layer_clone_bott:share(conv_layer_top, 'weight','bias')

end

function KernelNet:format( sample )

	--
	-- takes a sample of the form 31 x 3 and outputs 3 vectors 
	-- self.lookback x 1
	-- self.lookback x 120
	-- self.lookback x 6
	--
	local act_hist = build_hist(sample[{{},3}], self.act_count)
	local loc_hist = build_hist(sample[{{},2}], self.cluster_count)
	local impulses = sample:ne(0):t():clone():double()
	local val_input = {{sample[{{},1}]:clone():view(2*self.lookback + 1, 1):clone(), loc_hist, act_hist}, impulses}
 	local out = {val_input, impulses:clone()}
 	return out
end

function normalize( xs, mean, std )
	local xi = xs:ne(0)
	if xi:sum() == 0 then
		return xs, 0, 0
	end
	mean = mean or xs:sum() / xi:sum()
	std = std or torch.sqrt( (torch.pow(xs,2):sum() / xi:sum()) - (mean * mean))
	if xs:size(2) > 1 then
		mean = torch.mean(xs,2):repeatTensor(1,xs:size(2))
		std = torch.std(xs,2):repeatTensor(1,xs:size(2))
		xs = xs - mean
		local neq0 = std:ne(0)
		xs[neq0] = xs[neq0]:cdiv(std[neq0])		
	else
		xs = xs - mean
	end
	if xs:size(2) == 1 then
		xs = torch.cmul(xs, xi)
	else
		xs = torch.cmul(xs, mean:ne(0))
	end
	if xs:size(2) == 1 and std > 0 then
		xs = xs/std
	end
	return xs:clone():cuda(), mean, std
end

function cudafyTable( input )
	input[1][1][1] = input[1][1][1]:cuda()
	input[1][1][2] = input[1][1][2]:cuda()
	input[1][1][3] = input[1][1][3]:cuda()
	input[2] = input[2]:cuda()
end

function KernelNet:maskArray( t ) 
	t[1][1][1][{{self.mask - self.buffer, self.mask}}]:fill(0)
	t[1][1][2][{{self.mask - self.buffer, self.mask}}]:fill(0)
	t[1][1][3][{{self.mask - self.buffer, self.mask}}]:fill(0)
	
	t[1][2][{{1,3},{self.mask - self.buffer, self.mask}}] = 0
	t[2][{{1,3},{self.mask - self.buffer, self.mask}}]:fill(0)
end

function KernelNet:clipGrad( gradient )
	if gradient ~= gradient then
		gradient = 0
	elseif gradient < -1 * self.maxgrad  then
		gradient = -1 * self.maxgrad 
	elseif gradient > self.maxgrad  then
		gradient = self.maxgrad 
	end
	return gradient
end

function KernelNet:normalizeInput( input )
	--
	-- wrapper for normalize
	--
	-- returns only the mean and stddev of the blood glucose input,
	-- which is what we want to predict

	local enone, fnone, mean, std = 0
	input[1][1][1], mean, std = normalize(input[1][1][1])
	input[1][1][2], fnone, enone = normalize(input[1][1][2])
	input[1][1][3], fnone, enone = normalize(input[1][1][3])
	return mean, std
end

function KernelNet:updateTrainStats( mse )
	if( mse < 10000 and mse == mse) then 
		self.total_mse = self.total_mse + mse
		self.counter = self.counter + 1
	end
end

function KernelNet:updateValidStats( result, target )
	self.total_mse = self.total_mse + torch.pow(result - target, 2):squeeze()
	self.counter = self.counter + 1
end

function KernelNet:getRMSE()
	return math.sqrt(self.total_mse/self.counter)
end

function KernelNet:setupHyperParams(lookback, cluster_count, act_count, mask, buffer)
	self.minobs_valid = 100
	self.minobs_train = 10
	self.learning_rate = 1
	self.learning_rate_decay = 0.02
	self.lookback = lookback or 30
	self.cluster_count = cluster_count or 120
  	self.act_count = act_count or 6
  	self.buffer = buffer or 5
	self.minobs_bg_valid = 2 * self.lookback - (self.buffer + 1)

  	self.mask = mask or 2 * self.lookback + 1
    self.maxgrad = maxgrad or 2
end

function KernelNet:resetStats()
	self.total_mse = 0
	self.counter = 1
end

function KernelNet:updateVisuals( epoch )
	-- body
	--print("epoch " .. epoch ..  "training error: " .. total_mse/counter)
	--print(total_mse/counter, mseloss, output, target)	
	--print(conv_layer_clone_bott.weight)
	--gnuplot.plot({'bg temporal filter',conv_layer_top.weight[1][{{1,61}}]:float():squeeze(), '-'})
	gnuplot.plot({'bg temporal filter',conv_layer_top.weight[1][{{1,61}}]:float():squeeze(), '-'},{'location temporal filter',conv_layer_top.weight[1][{{62,122}}]:float():squeeze(), '-'}, {'activity temporal filter',conv_layer_top.weight[1][{{122,183}}]:float():squeeze(), '-'})
	--gnuplot.plot(self.cluster_kern:float():squeeze())
	--gnuplot.plot({'location weights', self.cluster_kern:float():squeeze(), '-'})	
end

function KernelNet:checkConditions( sample, input )
	local has_target = input[1][1][1][self.mask]:squeeze() ~= 0
	local has_recent = input[1][1][1][self.mask - (self.buffer + 1)]:squeeze() ~= 0
	local has_enough = sample:ne(0):sum() > self.minobs_valid
	local has_enough_bgs = input[1][1][1][{{1,self.mask - self.buffer}}]:ne(0):sum() > self.minobs_bg_valid
	return has_target and has_recent and has_enough_bgs 
end

function KernelNet:train( data, epoch )
	collectgarbage()
	local shuffle_idxs = torch.randperm(data:size(1) - 2*self.lookback):add(self.lookback)
	self:resetStats()
	for idx=1, data:size(1) - 2*self.lookback do
		
	--	if(idx % 1000 == 1) then self:updateVisuals() end
		local i = shuffle_idxs[idx] 
		local sample = data[{{i - self.lookback, i + self.lookback}}]
		local input = self:format(sample)
		local target = input[1][1][1][self.mask]:clone()
		if target[1] ~= 0 and sample:ne(0):sum() > self.minobs_train then
			self:zeroGradParameters()	
			self:maskArray(input)
			cudafyTable(input)
			mean, std = KernelNet:normalizeInput(input)
			target = (target - mean) / std

			local output = self:forward(input)
			
			target = target:viewAs(output):cuda()
		
			local mseloss = self.criterion:forward(output:cuda(), target:cuda())

			local gradient = self.criterion:backward(output:cuda(), target:cuda())
			self:updateTrainStats(mseloss)
			gradient[1][1] = self:clipGrad(gradient[1][1])
			self:backward(input, gradient)
			local current_learning_rate = self.learning_rate * self.learning_rate_decay
			self:updateParameters(current_learning_rate)

		end
	end
	print("epoch " .. epoch .. " got " .. self.total_mse / self.counter .. " MSE on " .. self.counter .." training samples")
end

function KernelNet:predict( data )
	-- method used for testing and validation
	
	self:resetStats()

	for i=self.lookback + 1, data:size(1) - self.lookback do
		local sample = data[{{i - self.lookback, i + self.lookback}}]
		local input = self:format(sample)
		local target = input[1][1][1][self.mask]:clone()
		if self:checkConditions(sample,input) then

			self:maskArray(input)
			cudafyTable(input)
			mean, std = KernelNet:normalizeInput(input)

			local output = self:forward(input)
			local result = (output * std) + mean		
			self:updateValidStats(result, target)
		end
	end
	print("got -|" .. self:getRMSE() .. "|- RMSE on -|" .. self.counter .."|- validation samples")
	return self:getRMSE()
end

local args = {...} 


function test()
	local filename = os.date("%c", os.time()):gsub(' ','-') .. ".net" 
	local test_net = KernelNet:new()
	test_net = test_net:cuda()
	x = assert(loadfile('readjson.lua'))(args[1])
	print(x:size())
	print(test_net)
	local t_split = 5.0 / 6.0
	local tv_split = 4.0 / 5.0

	x = x[{{1, x:size(1) * t_split}}]:clone()
	local training_data = x[{{1,x:size(1) * tv_split}}]:clone()
	local validation_data = x[{{x:size(1) * tv_split, x:size(1)}}]:clone()
	--test_net:predict(validation_data)
	local valid_score = 10000000
	for i=1,10 do
		test_net:train(training_data, i)
		local vs = test_net:predict(validation_data)
		if vs < valid_score then
			valid_score = vs 
			torch.save("experiments/" .. filename , test_net)
		end
		test_net:updateVisuals()
	end
end

test()



