
require 'torch'
require 'math'
require 'nn'
require 'kernelutil'
require 'CDivTable_rebust'
require 'MSETVCriterion'
require 'gnuplot'
require 'debug'
require 'os'
local json = require('cjson')


-- local Categorical, parent = torch.class('nn.Categorical', 'nn.DepthConcat')

-- KernelNet = {}
-- function Categorical:__init (category_count, time_slices)
  
--   --
--   --	input: tensor of size <time_slices> * <category_count>
--   --	output: tensor of size <time_slices>
--   --
--   --	weights and bias of category_kernel_0 are shared across each time step, and learn
--   -- 	a time independent importance weighting
--   --
--   parent.__init(self,3)

--  -- o = nn.DepthConcat(1)
--   self.category_kernel_0 =  nn.Linear(category_count, 1)
--   self:add(category_kernel_0)

--   for i=1,time_slices do
--   	local kernel_copy = self.category_kernel_0:clone('weight', 'bias')
--   	self:add(kernel_copy)
--   end
-- end

-- function Categorical:updateOutput(input)
-- 	print(self)
-- 	print(self.output)
-- 	print("asdas")
-- 	return self.output
-- end

KernelNet, parent = torch.class('nn.KernelNet', 'nn.ParallelTable')

function KernelNet:__init (lookback, cluster_count, act_count, mask, buffer, maxgrad)
	parent.__init(self)  
	self:initq(lookback, cluster_count, act_count, mask, buffer)
end

function KernelNet:initq(lookback, cluster_count, act_count, mask, buffer)
  
  main_seq = nn.Sequential()
  self:setupHyperParams(lookback,cluster_count,act_count,mask,buffer)
  self.criterion = nn.MSETVCriterion()
  kernel_matrix_init = torch.Tensor((self.lookback*2+1) * 3):fill(1)
  kernel_matrix_init[{{62,122}}]:fill(1)
  kernel_matrix_init[{{122,183}}]:fill(1)

  temporal_ratio = nn.ParallelTable()
  self.conv_layer_top = nn.Linear((self.lookback*2+1) * 3,1)
  self.conv_layer_top.weight = kernel_matrix_init:clone():viewAs(self.conv_layer_top.weight)
  self.conv_layer_top.bias = torch.Tensor({0}):viewAs(self.conv_layer_top.bias)
  conv_layer_clone_bott = self.conv_layer_top:clone('weight','bias')	
   
  all_vars = nn.ParallelTable()	       -- input is {[61 x 1], [61 x 120], [61 x 6]}  
  all_vars:add(nn.Identity())		   -- blood glucose vals (pass directly to temporal step)
  cluster_kern = nn.Linear(self.cluster_count, 1)
  act_kern = nn.Linear(self.act_count, 1)

  -- act_kern.weight = act_kern.weight:fill(1)
  -- cluster_kern.weight = cluster_kern.weight:fill(1)

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

  top_bottom_regularizers = nn.ParallelTable()
  top_bottom = nn.ParallelTable()
  top_bottom:add(zero_out)
  top_bottom:add(all_impulse_holder)


  main_seq:add(top_bottom)


  temporal_ratio:add(self.conv_layer_top)
  temporal_ratio:add(conv_layer_clone_bott)

  main_seq:add(temporal_ratio) -- expects {Tensor(1,1,(lookback*2+1,3), Tensor(1,1,(lookback*2+1,3)}
  main_seq:add(CDivTable_robust())
  self:add(main_seq)
  l1_regularizer = nn.CMul(self.cluster_kern:size())
  
  l2_act_time_regularizer = nn.CMul(self.conv_layer_top.weight:size())
  l2_loc_time_regularizer = nn.CMul(self.conv_layer_top.weight:size())
  l2_bg_time_regularizer = nn.CMul(self.conv_layer_top.weight:size())
  l2_act_regularizer = nn.CMul(self.act_kern:size())

  self:add(l1_regularizer)
  self:add(l2_act_time_regularizer)
  self:add(l2_loc_time_regularizer)
  self:add(l2_bg_time_regularizer)
  self:add(l2_act_regularizer)

  l1_regularizer:share(cluster_kern, 'weight')
  l2_act_time_regularizer:share(self.conv_layer_top, 'weight')
  l2_loc_time_regularizer:share(self.conv_layer_top, 'weight')
  l2_bg_time_regularizer:share(self.conv_layer_top, 'weight')
  l2_act_regularizer:share(act_kern, 'weight')

  conv_layer_clone_bott:share(self.conv_layer_top, 'weight','bias')

end

function KernelNet:format( sample )
	--
	-- takes a sample of the form {31 x 1 and outputs 3 vectors 
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

function KernelNet:critEval( output, target )

	local l1_zs = self.cluster_kern:clone():fill(0)
    local l2_zs = self.conv_layer_top.weight:clone():fill(0)
    local l2_act_zs = self.act_kern:clone():fill(0)

	local target_table = {target, l1_zs, l2_zs, l2_zs, l2_zs, l2_act_zs}
	
	local mseloss = self.criterion:forward(output, target_table)

	local gradient = self.criterion:backward(output, target_table)
	return gradient, mseloss

end

function normalize( xs, mean, std )
	local xi = xs:ne(0):double()
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
		xs = torch.cmul(xs, mean:ne(0):double())
	end
	if xs:size(2) == 1 and std > 0 then
		xs = xs/std
	end
	return xs:clone(), mean, std
end

function cudafyTable( input )
	input[1][1][1] = input[1][1][1]:double()
	input[1][1][2] = input[1][1][2]:double()
	input[1][1][3] = input[1][1][3]:double()
	input[2] = input[2]:double()
end

function KernelNet:maskArray( t ) 
	t[1][1][1][{{self.mask - self.buffer, self.mask}}]:fill(0)
	t[1][1][2][{{self.mask - self.buffer, self.mask}}]:fill(0)
	t[1][1][3][{{self.mask - self.buffer, self.mask}}]:fill(0)
	
	t[1][2][{{1,3},{self.mask - self.buffer, self.mask}}] = 0
	t[2][{{1,3},{self.mask - self.buffer, self.mask}}]:fill(0)
	if self.hide_exogenous then
		t[1][1][2]:fill(0)
		---t[1][1][3]:fill(0)
		t[1][2][{{2,2},{1, self.mask}}] = 0
		t[2][{{2,2},{1, self.mask}}]:fill(0)
	end
end

function KernelNet:clipGrad( gradient, maxgrad )
	local maxgrad = maxgrad or self.maxgrad
	if gradient ~= gradient then
		gradient = 0
	elseif gradient < (-1 * maxgrad)  then
		gradient = -1 * maxgrad 
	elseif gradient > maxgrad  then
		gradient = maxgrad 
	end
	return gradient
end

function KernelNet:clipGradTensor(gradient, maxgrad)
	 gradient[gradient:gt(maxgrad)] = maxgrad
	 gradient[gradient:lt(-1 *maxgrad)] =  -1 * maxgrad
	 return gradient
end

function KernelNet:clipBias( )
	for k,v in pairs({self.conv_layer_top, act_kern, cluster_kern}) do
		v.bias[1] = self:clipGrad(v.bias[1], self.maxbias)
	end
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
	self.total_mse = self.total_mse + ((result - target:squeeze()) * (result - target:squeeze())):squeeze()
	self.counter = self.counter + 1
end

function KernelNet:getRMSE()
	return math.sqrt(self.total_mse/self.counter)
end

function KernelNet:setupHyperParams(lookback, cluster_count, act_count, mask, buffer)
	self.minobs_valid = 60
	self.minobs_train = 12
	self.learning_rate = 2
	self.hide_exogenous = false
	self.learning_rate_decay = 0.01
	self.tv_lambda = 0.05
	self.lookback = lookback or 30
	self.cluster_count = cluster_count or 120
	self.cluster_redundancy = 8 
  	self.act_count = act_count or 6
  	self.buffer = buffer or 5
	self.minobs_bg_valid = 2 * self.lookback - (self.buffer + 1)
	self.minobs_bg_train = 0.3 * self.lookback 
  	self.mask = mask or 2 * self.lookback + 1
    self.maxgrad = maxgrad or 2
    self.maxbias = 0.005
    self.total_mse = 0
end

function KernelNet:resetStats()
	self.total_mse = 0
	self.counter = 1
end

function KernelNet:updateVisuals( epoch )

	local bg_time = {'bg temporal filter',self.conv_layer_top.weight[1][{{1,61}}]:float():squeeze(), '-'}
	local act_time = {'location temporal filter',self.conv_layer_top.weight[1][{{62,122}}]:float():squeeze()*3, '-'}
	local loc_time = {'activity temporal filter',self.conv_layer_top.weight[1][{{123,183}}]:float():squeeze()*3, '-'}
	local clusters = {'cluster weights', torch.linspace(1,30, self.cluster_kern:nElement()), self.cluster_kern:float():squeeze() + 6, '-'}

	local stretched = self.act_kern:float():squeeze()
	local act_s = {'activity weights', torch.linspace(1,20, stretched:nElement()), stretched + 12, '-'}

	gnuplot.plot(bg_time, act_time, loc_time, act_s, clusters)
	--gnuplot.plot({'location weights', self.cluster_kern:float():squeeze(), '-'})	
end

function KernelNet:checkConditions( sample, input )
	local has_target = input[1][1][1][self.mask]:squeeze() ~= 0
	local has_recent = input[1][1][1][self.mask - (self.buffer + 1)]:squeeze() ~= 0
	local has_enough = sample:ne(0):sum() > self.minobs_valid
	local has_enough_bgs = input[1][1][1][{{1,self.mask - self.buffer}}]:ne(0):sum() > self.minobs_bg_valid
	return has_target and has_recent and has_enough_bgs 
end

function KernelNet:totalVariation()
	-- body
	local w_bg = self.conv_layer_top.weight[1][{{1,2*self.lookback + 1}}]:squeeze()
	local offset = torch.Tensor(2*self.lookback + 2):fill(0)
	offset[{{2,2*self.lookback + 2}}] = w_bg
	return (w_bg - offset):abs():sum()
end

function KernelNet:makeTimeReg(name)
	local out = self.conv_layer_top.weight:clone():fill(0)
	if name == 'act' then
		out[{1,{123,183}}] = 1
	elseif name == 'loc' then
		out[{1,{62,122}}] = 1
	elseif name ==  'bg' then
		out[{1,{1,61}}] = 1
	end
	return out
end

function KernelNet:train( data, epoch )
	collectgarbage()
	local shuffle_idxs = torch.randperm(data:size(1) - 2*self.lookback):add(self.lookback)
	self:resetStats()
	local output = nil
	local zeroes = 0
	for idx=1, data:size(1) - 2*self.lookback do
		
	  --  if(idx % 500 == 1) then self:updateVisuals() end
		local i = shuffle_idxs[idx] 
		local sample = data[{{i - self.lookback, i + self.lookback}}]
		local input = self:format(sample)
		local target = input[1][1][1][self.mask]:clone()

		local cond1 = sample[{{},1}]:ne(0):sum() > self.minobs_bg_train 
		if target[1] ~= 0 
			and cond1 and (sample[{{},2}]:ne(0):sum() > self.minobs_train or sample[{{},3}]:ne(0):sum() > self.minobs_train) then
			self:zeroGradParameters()	
			self:maskArray(input)
			cudafyTable(input)
			mean, std = KernelNet:normalizeInput(input)
			target = (target - mean) / std
			local new_input = {input, self.cluster_kern:clone():fill(1), self:makeTimeReg('act'), self:makeTimeReg('loc'), self:makeTimeReg('bg'), self.act_kern:clone():fill(1)}
			output = self:forward(new_input)
		
			if output[1]:squeeze() == 0 then zeroes = zeroes + 1 end

			local gradient, mseloss = self:critEval(output, target)
			self:updateTrainStats(mseloss)
			gradient[1][1][1] = self:clipGrad(gradient[1][1][1])
			gradient[2] = self:clipGradTensor(gradient[2], self.maxgrad)
			gradient[3] = self:clipGradTensor(gradient[3], self.maxgrad)
			gradient[4] = self:clipGradTensor(gradient[4], self.maxgrad)
			gradient[5] = self:clipGradTensor(gradient[5], self.maxgrad)
			gradient[6] = self:clipGradTensor(gradient[6], self.maxgrad)


			self:backward(new_input, gradient)
			local current_learning_rate = self.learning_rate  * self.learning_rate_decay
			self:updateParameters(current_learning_rate)
			self:clipBias()

		end
	end
	print("epoch " .. epoch .. " got " .. self.total_mse / self.counter .. " MSE on " .. self.counter .." training samples: percent 0s: " .. zeroes/self.counter)
end

function KernelNet:predict( data )
	-- method used for testing and validation
	local total_mse_static = 0
	self:resetStats()
	for i=self.lookback + 1, data:size(1) - self.lookback do
		local sample = data[{{i - self.lookback, i + self.lookback}}]
		local input = self:format(sample)
		local target = input[1][1][1][self.mask]:clone()
		local static_guess = input[1][1][1][self.mask - (self.buffer + 1)]
		if self:checkConditions(sample,input) then

			self:maskArray(input)
			cudafyTable(input)
			mean, std = KernelNet:normalizeInput(input)
			local new_input = {input, self.cluster_kern:clone():fill(1), self:makeTimeReg('act'), self:makeTimeReg('loc'), self:makeTimeReg('bg'), self.act_kern:clone():fill(1)}
			local output = self:forward(new_input)[1]

			local result = (output * std) + mean		
			self:updateValidStats(result, target)
			total_mse_static = total_mse_static + (static_guess[1] - target[1])^2
		end
	end
	print("got -|" .. self:getRMSE() .. "|- RMSE on -|" .. self.counter .."|- validation samples vs " .. math.sqrt(total_mse_static/self.counter))
	return self:getRMSE()
end

local args = { ... } 


function KernelNet:saveWeights(filename)
	local json_data = {location_weights=self.cluster_kern:clone():double():totable(), 
					   act_weights=self.act_kern:clone():double():totable(), 
					   temporal_weights=conv_layer_top.weight:clone():double():totable()}
	local json_str = json.encode(json_data)
	local f = assert(io.open("experiments/weights" .. filename .. ".json", "w+"))
    	local t = f:write(json_str)
	f:close()
end

function KernelNet:save( filename )
	-- body
	torch.save("experiments/" .. filename .. ".t7", self:clone('weight','bias'))
	self:saveWeights( filename )
end


