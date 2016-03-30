require 'torch'
require 'nn'
require 'gnuplot'
require 'kernelutil'
require 'visualizer'
require 'CDivTable_rebust' 
local json = require('cjson')


local ScalarOneHot, parent = torch.class('nn.ScalarOneHot', 'nn.Sequential')

function savejsonfile( data, filename )
	print("saving to " .. filename)
	local json_str = json.encode(data)
	print("garbage collection")
	collectgarbage()
	local f = assert(io.open( filename , "w+"))
	f:write(json_str)
	f:close()
end


function loadjson_asarray( filename )
	print(filename)
	local f = assert(io.open(filename))
	local json_str = f:read("*all")
	f:close()
	local datatable = json.decode(json_str)

	local out = torch.Tensor(datatable)
	return out
end




function ScalarOneHot:__init(scalar_count, onehot_count, weights)
	-- body
    parent.__init(self)
   	weights = weights or nil
	self.onehot_count = onehot_count or 50
	self.target_types = 6
	self.scalar_count = scalar_count or 1
	self.range = range or (42)  -- 3.5 hrs
	self.min_obs = (210 / 5) - 1
	self.maxgrad = 1
	self.bg_spacing = 20
	self.learning_rate = 50
	self.learning_rate_decay = 0.001
	self.see_future = false
	self:zeroStats()

	self.averaging_pd = 11
	self.criterion = nn.MSECriterion()

	-- two main tables to hold different variable types 
	--
	-- input should look like: 
	--  {
	--		{scalar_values (numerator), onehot_values (numerator)},
	--		{impuse_values (denominator), onehot_normalizer (denominator)}
	--	}
	--

	self.container = nn.ParallelTable()

	-- each one must be split into the numerator and denominator
	self.numerator = nn.Sequential()
	self.numer_table = nn.ParallelTable()

	local sckern_sz = self.scalar_count * (2 * self.range + 1);
	local ohkern_sz = self.onehot_count * (2 * self.range + 1)
	self.top_scalar = nn.Linear(self.scalar_count * (2 * self.range), 1)
	self.top_scalar.weight:fill(1)
	self.top_onehot = nn.SparseLinear(self.onehot_count * (2 * self.bg_spacing * self.range + 1), 1)
	

	self.numer_table:add(self.top_scalar)
	self.numer_table:add(self.top_onehot)
	self.numerator:add(self.numer_table)
	self.numerator:add(nn.CAddTable())
	self.container:add(self.numerator)

	self.denominator = nn.Sequential()
	self.denom_table = nn.ParallelTable()
	self.bottom_scalar = self.top_scalar:clone('weight','bias')
	self.bottom_scalar:share(self.top_scalar,'weight', 'bias')
	self.denom_table:add(self.bottom_scalar)

	self.denom_table:add(nn.Identity())
	self.denominator:add(self.denom_table)
	self.denominator:add(nn.CAddTable())
	self.container:add(self.denominator)

	self:add(self.container)
	self:add(CDivTable_robust())
end

function ScalarOneHot:format(sample)
	--
	-- expects a tensor [range] x 12 
	-- last column is the categorical (onehot) variable we are interested in
	-- second column is scalar (bg)
	-- cols 5 - 11 are activity scalars
	--

	local count = sample:ne(0):sum()
	local out = torch.Tensor(math.max(count,1) , 2)
	local i = 1
	-- the rest
	out[1] = torch.Tensor({1,0})
	for j=1, sample:size(1) do
		local cluster = sample[j]
		if( cluster ~= 0 ) then
			--print(cluster)
			out[i] = torch.Tensor({cluster * j + j, 1})
			i = i + 1
		end
	end
	return out
end

function ScalarOneHot:normalize( sample, mean, std )
	-- body
	local out = sample
	local neq0 = out:ne(0)  
	mean = mean or torch.mean(out[neq0])
	std = std or torch.std(out[neq0])
	if ( std == 0 or std ~= std ) then
		std = 1
	end

	local output = out:clone():fill(0)

	local nnzs = out[neq0]:clone()
	out[neq0] = (nnzs - mean) / std

	return out, mean, std
end

function ScalarOneHot:format_scalar( sample, mean, std )
	local out = torch.zeros(self.range * 2)
	for i=0, out:size(1) - 1 do
		out[i + 1] = sample[(i * 20) + 1]
	end
	return out
end

function ScalarOneHot:batchFormat( data )
	local padding = self.range
	local padded_data = pad(data, padding)
	local out = torch.Tensor(data:size(1), 2* padding + 1, self.onehot_count)
	for i=1,data:size(1) do
		out[i] = self:format(padded_data[{{i, i + (2 * padding)}}])
	end
	return out
end

function ScalarOneHot:updateVisuals(  )
	-- body
	gnuplot.figure(1)
	local top = self.top_scalar.weight[{1,{}}]:view(2*self.range):squeeze()

	

	local bottom = self.bottom_scalar.weight[{1,{}}]:view(2*self.range)
	
	local sparse_data = self.top_onehot.weight[{1,{}}]:view(2*self.range * self.bg_spacing + 1,self.onehot_count)

	local avgs1 = torch.mean(torch.abs(sparse_data),1)
	local avgs2 = torch.mean(sparse_data,2)
	gnuplot.plot({'thai', sparse_data[{{},43}], '-'}) --, {'15',sparse_data[{{},15}], '-'})
	--gnuplot.plot( {'1',sparse_data[{{},1}], '-'}, {'12',sparse_data[{{},12}], '-'}, {'10', sparse_data[{{},10}], '-'}, {'44',sparse_data[{{},44}], '-'} )
	--
	--gnuplot.raw('set multiplot layout 2,1')
	--draw_onehot(data[{{2500, 12000}}], self)
	--draw_onehot_nll(data[{{2500, 12000}}], self)
	--gnuplot.raw('unset multiplot')
end

function ScalarOneHot:updateTrainStats( mse )
	self.total_mse = self.total_mse + mse
	self.counter = self.counter + 1
end

function ScalarOneHot:clipGradTensor(gradient, maxgrad)
	 gradient[gradient:gt(maxgrad)] = maxgrad
	 gradient[gradient:lt(-1 *maxgrad)] =  -1 * maxgrad
	 gradient[gradient:ne(gradient)] = 0
	 return gradient
end


function ScalarOneHot:updateAccuracy( input, target, i )
	-- body
	self.counter = self.counter + 1
	self.total_correct = input == target and self.total_correct + 1 or self.total_correct
	self.accuracy_stats[target] = input == target and self.accuracy_stats[target] + 1 or self.accuracy_stats[target]
	self.actual_totals[target] = self.actual_totals[target] + 1
	self.predicted_totals[input] = self.predicted_totals[input] + 1
end

function ScalarOneHot:zeroStats() 
	self.total_correct = 0
	self.counter = 0
	self.accuracy_stats = torch.Tensor(self.onehot_count):fill(0)
	self.actual_totals = torch.Tensor(self.onehot_count):fill(0)
	self.predicted_totals = torch.Tensor(self.onehot_count):fill(0)
end

function ScalarOneHot:statsToString( )
	local out = "total accuracy: " .. self.total_correct/self.counter .. " per class accuracy: "
	for k,v in pairs(self.accuracy_stats:totable()) do
		local precision = v / self.predicted_totals[k]
		local recall = v / self.actual_totals[k]
		out = out .. "\n" .. k .. " - " .. v/self.actual_totals[k] .. " f1: " .. 2 * precision * recall / (precision + recall)
	end
	return out:sub(1,-3)
end

function ScalarOneHot:nearestGuessEval( data )

	-- method used for benchmarking

	self:zeroStats()
	local obs = self.see_future and 1 or 2

	for i= self.range + 1, data:size(1) - self.range do
		local sample = data[{{i - self.range, i + self.range}}]:clone()
		local target = sample:clone()[obs * self.range + 1]
		sample[obs * self.range + 1] = 0
		if target ~= 0 and sample:ne(0):sum() > self.min_obs then
			target = target_keys[target]
			local nid = nearestIndex(sample, obs * self.range + 1)
			local guess = target_keys[sample[nid]]
			self:updateAccuracy(guess, target)
		end
	end
	print("nearest neighbor imputer on validation samples: ")
	print(self:statsToString())
	return self.total_correct / self.counter
end

function ScalarOneHot:modeEval( data )

	-- method used for benchmarking

	self:zeroStats()
	for i=self.range + 1, data:size(1) - self.range do
		local sample = data[{{i - self.range, i + self.range}}]:clone()
		local obs = self.see_future and 1 or 2

		local target = sample:clone()[obs * self.range + 1]
		sample[obs * self.range + 1] = 0
		if target ~= 0 and sample:ne(0):sum() > self.min_obs then
			target = target_keys[target]
			local counts = torch.Tensor({sample:eq(1):sum(),sample:eq(2):sum(),sample:eq(3):sum(),
										sample:eq(4):sum(),sample:eq(5):sum(),sample:eq(6):sum()})
			self:updateAccuracy(target_keys[argmax(counts)[1]], target)
		end
	end
	print("mode imputer on validation samples: ")
	print(self:statsToString())
	return self.total_correct / self.counter
end



function ScalarOneHot:valid( data )
	
	-- method used for validation

	local total_loss = 0
	self:zeroStats()
	local obs = self.see_future and 1 or 2

	for i=self.range + 1, data:size(1) - self.range do
		local sample = data[{{i - self.range, i + self.range}}]:clone()
		local target = sample:clone()[obs * self.range + 1]
		sample[obs * self.range + 1] = 0
		local input = self:format(sample)
		if target ~= 0 and sample:ne(0):sum() > self.min_obs then
			local output = self:forward(input)

			total_loss = total_loss + self.criterion:forward(output, target)
			local softmax = torch.exp(output)
			idx = target_keys[argmax(softmax)[1]]
			self:updateAccuracy(idx, target)
		end
	end
	print("got -|" .. total_loss / self.counter .. "|- RMSE on -|" .. self.counter .."|- validation samples")
	print(self:statsToString())
	return total_loss / self.counter
end

function ScalarOneHot:validateConfidence( data )
local total_loss = 0
	self:zeroStats()
	local obs = self.see_future and 1 or 2

	for i=self.range + 1, data:size(1) - self.range do
		local sample = data[{{i - self.range, i + self.range}}]:clone()
		local target = sample:clone()[obs * self.range + 1]
		sample[obs * self.range + 1] = 0
		local input = self:format(sample)

		if target ~= 0 and sample:ne(0):sum() > self.min_obs then
			target = target_keys[target]
			local output = self:forward(input)
			total_loss = total_loss + self.criterion:forward(output, target)
			local softmax = torch.exp(output)
			idx = target_keys[argmax(softmax)[1]]
			self:updateAccuracy(idx, target)
		end
	end
	print("got -|" .. total_loss / self.counter .. "|- RMSE on -|" .. self.counter .."|- validation samples")
	print(self:statsToString())
	return total_loss / self.counter
end

function ScalarOneHot:batchProb( data )
	
	-- returns a tensor of log probabilities for each class for each sample 
	assert(self.see_future == true)
	local out = torch.Tensor(data:size(1), self.onehot_count):fill(0)
	for i=self.range + 1, data:size(1) - self.range do
		local sample = data[{{i - self.range, i + self.range}}]:clone()
		local input = self:format(sample)
		if input:sum() > 0 then
			out[i] = torch.exp(self:forward(input))
		else
			out[i] = torch.exp(self:forward(input)):fill(0)
		end
	end

	-- local period = 5;
	-- local avg = out:clone():fill(0)

	-- for i=1,out:size(1) do
	-- 	local left = math.max(1, i - period/2)
	-- 	local right = math.min(out:size(1), i + period/2)
	-- 	avg[i] = out[{{left,right}}]:mean(1)
	-- end

	return out
end

function ScalarOneHot:group_fmt( data , i )
	-- body
end

function ScalarOneHot:train( data_scalar, data_onehot, n, epoch_size )

	-- performs one epoch of training 

	batch_size = epoch_size or 1
	collectgarbage()
	self.total_mse = 0
	self.counter = 0
	local obs = self.see_future and 1 or 2
	local locscnt = 0
	local nonzeros = data_scalar:nonzero()

	nonzeros = nonzeros[nonzeros:gt(2 * self.range * 20)]

	local shuffle_idxs = torch.randperm(nonzeros:size(1))
	print(math.floor(shuffle_idxs:size(1) * batch_size))
	for idx=1, math.floor(shuffle_idxs:size(1) * batch_size) do
		local i = nonzeros[shuffle_idxs[idx]] - (self.range * 20)

		local onehot_sample = data_onehot[{{i - 20 * self.range, i + 20 *self.range}}]:clone()
		local scalar_sample = data_scalar[{{i - 20 * self.range, i + 20 *self.range}}]:clone()
		local target = scalar_sample:clone()[20 * obs * self.range + 1]
		scalar_sample[obs * self.range + 1] = 0
		--scalar_sample = augment_time(scalar_sample, 1)
		onehot_sample = augment_time(onehot_sample, 1)

		-- print(scalar_sample)

		scalar_sample = self:format_scalar(scalar_sample)
		-- print(scalar_sample)
		if target ~= 0 and scalar_sample:ne(0):sum() > self.min_obs then
			local scalar_input, mean, std = self:normalize(scalar_sample)
			target = (target - mean) / std


			--onehot_sample:fill(0)
			local onehot_input = self:format(onehot_sample)
			local input_table = {
									{scalar_input:double(), onehot_input:double()}, 
									{scalar_sample:clone():ne(0):double(), torch.Tensor({onehot_sample:clone():ne(0):sum()})}
								}
			
			locscnt = (onehot_sample:sum() > 0) and locscnt + 1 or locscnt
			local output = self:forward(input_table)
			target = torch.Tensor({target})
			local mseloss = self.criterion:forward(output, target)
			self:updateTrainStats(mseloss * std)
			self:zeroGradParameters()
			local gradient = self.criterion:backward(output, target)
			gradient = self:clipGradTensor(gradient, self.maxgrad)
			self:backward(input_table, gradient)
			local current_learning_rate = self.learning_rate  * self.learning_rate_decay
			self:updateParameters(current_learning_rate)
		end
	end
	self:updateVisuals()
	print("epoch " .. n .. " got " .. self.total_mse / self.counter .. " MSE on " .. self.counter .." training samples and " .. locscnt)
end

function test()
	local net = nn.ScalarOneHot(1,80)
	print(net)

	local data = loadjson_asarray( arg[1] )
	print(data:size())
	local td, vv, vd = tts(data) 
	
	td = td:cat(vv, 1):cat(vd, 1)

	print("training_data size " .. td:size(1))
	print("validation_data size " ..  vd:size(1))

	training_data_onehot = td[{{},td:size(2)}]
	valid_data_onehot = vd[{{},vd:size(2)}]

	training_data_scalar_bgs = td[{{},2}]
	valid_data_scalar_bgs = vd[{{},2}]

	local num_acts = 6
	for i=1,100 do
		net:train(training_data_scalar_bgs, training_data_onehot, i)
		--net:updateVisuals(valid_data_scalar_bgs, valid_data_onehot)

	--	local loss = net:valid(valid_data_scalar_bgs, valid_data_onehot)
		--net:nearestGuessEval(valid_data)
		--net:modeEval(valid_data)

		-- print(net.actual_totals)
		-- if loss < min_loss then
		-- 	min_loss = loss
		-- 	torch.save("activity_kernel.t7", net)
		-- 	print("network saved")
		-- 	-- look at one week
		-- end
	end
end

function apply_model( model_fn, data_fn )
	local net = torch.load(model_fn)
	local training_data, valid_data = train_test_split()
	local data = training_data:cat(valid_data, 1)
	local out = net:batchProb(data[{{},data:size(2)}])
	print("got here")

	local new_data = data:cat(out)
	return new_data
end


if arg[1] == 'apply' then
	local new_data = apply_model(arg[2] or 'data/activity_kernel_aug_49_15sec.t7', '')
	local data_table = new_data:totable()
	savejsonfile(data_table, 'data/fulldata_w_likelyhoods.json')
else
	test()
end

