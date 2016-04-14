require 'torch'
require 'nn'
require 'gnuplot'
require 'kernelutil'
require 'visualizer' 
local json = require('cjson')


local ActivityImputer, parent = torch.class('nn.ActivityImputer', 'nn.Sequential')

function savejsonfile( data, filename )
	print("saving to " .. filename)
	local json_str = json.encode(data)
	print("garbage collection")
	collectgarbage()
	local f = assert(io.open( filename , "w+"))
	f:write(json_str)
	f:close()
end


function ActivityImputer:__init(act_types, weights)
	-- body
    parent.__init(self)
   	weights = weights or nil
	self.act_types = act_types or 6
	self.target_types = 6
	self.range = range or 40
	self.min_obs = 0
	self.maxgrad = 500000
	self.learning_rate = 0.6
	self.learning_rate_decay = 0.01
	self.see_future = false
	self:zeroStats()

	self.averaging_pd = 11
	self.criterion = nn.ClassNLLCriterion()
	self.kernel = nn.Linear(self.act_types * (2 * self.range + 1),self.target_types)
	self:add(nn.Reshape(self.act_types * (2 * self.range + 1)))
	
	self:add(self.kernel)
	self:add(nn.LogSoftMax())
end

function ActivityImputer:format(sample)
	--
	-- expects a tensor [range] x 1 of labels
	--
	local out = torch.Tensor(self.act_types,2*self.range + 1)
	out[{self.act_types - 1,{}}] = sample[{{}, sample:size(2) - 1}]:eq(1)
	out[{self.act_types,{}}] = sample[{{}, sample:size(2) - 1}]:gt(1)
	for i=1,self.act_types do
		if i < 7 then -- last "act" is really the speed
			out[{i,{}}] = sample[{{}, sample:size(2)}]:eq(i)
		end
	end
	
	return out
end

function ActivityImputer:batchFormat( data )
	local padding = self.range
	local padded_data = pad(data, padding)
	local out = torch.Tensor(data:size(1), 2* padding + 1, self.act_types)
	for i=1,data:size(1) do
		out[i] = self:format(padded_data[{{i, i + (2 * padding)}}])
	end
	return out
end

function ActivityImputer:updateVisuals( data )
	-- body
	gnuplot.figure(1)
	gnuplot.splot(self.kernel.weight[{1,{}}]:view(self.act_types,2*self.range + 1))
	--gnuplot.raw('set multiplot layout 2,1')
	--draw_onehot(data[{{6600, 6600+3500}}], self)
	--draw_onehot_nll(data[{{6600, 6600+3500}}], self)
	--gnuplot.raw('unset multiplot')
end

function ActivityImputer:updateTrainStats( mse )
	self.total_mse = self.total_mse + mse
	self.counter = self.counter + 1
end

function ActivityImputer:clipGradTensor(gradient, maxgrad)
	 gradient[gradient:gt(maxgrad)] = maxgrad
	 gradient[gradient:lt(-1 *maxgrad)] =  -1 * maxgrad
	 gradient[gradient:ne(gradient)] = 0
	 return gradient
end


function ActivityImputer:updateAccuracy( input, target, i )
	-- body
	self.counter = self.counter + 1
	self.total_correct = input == target and self.total_correct + 1 or self.total_correct
	self.accuracy_stats[target] = input == target and self.accuracy_stats[target] + 1 or self.accuracy_stats[target]
	self.actual_totals[target] = self.actual_totals[target] + 1
	self.predicted_totals[input] = self.predicted_totals[input] + 1
end

function ActivityImputer:zeroStats() 
	self.total_correct = 0
	self.counter = 0
	self.accuracy_stats = torch.Tensor(self.act_types):fill(0)
	self.actual_totals = torch.Tensor(self.act_types):fill(0)
	self.predicted_totals = torch.Tensor(self.act_types):fill(0)
end

function ActivityImputer:statsToString( )
	local out = "total accuracy: " .. self.total_correct/self.counter .. " per class accuracy: "
	for k,v in pairs(self.accuracy_stats:totable()) do
		local precision = v / self.predicted_totals[k]
		local recall = v / self.actual_totals[k]
		out = out .. "\n" .. k .. " - " .. v/self.actual_totals[k] .. " f1: " .. 2 * precision * recall / (precision + recall)
	end
	return out:sub(1,-3)
end

function ActivityImputer:nearestGuessEval( data )

	-- method used for benchmarking

	self:zeroStats()
	local obs = self.see_future and 1 or 2

	for i= self.range + 1, data:size(1) - self.range do
		local sample = data[{{i - self.range, i + self.range},2}]:clone()
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

function ActivityImputer:modeEval( data )

	-- method used for benchmarking

	self:zeroStats()
	for i=self.range + 1, data:size(1) - self.range do
		local sample = data[{{i - self.range, i + self.range}}]:clone()
		local obs = self.see_future and 1 or 2

		local target = sample:clone()[obs * self.range + 1][2]
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



function ActivityImputer:valid( data )
	
	-- method used for validation

	local total_loss = 0
	self:zeroStats()
	local obs = self.see_future and 1 or 2

	for i=self.range + 1, data:size(1) - self.range do
		local sample = data[{{i - self.range, i + self.range}}]:clone()
		local target = sample:clone()[obs * self.range + 1][2]
		sample[obs * self.range + 1] = 0
		local input = self:format(sample)
		if target ~= 0 and sample:ne(0):sum() > self.min_obs then
			target = target_keys[target]
			local output = self:forward(input)
			total_loss = total_loss + self.criterion:forward(output, target)
			--local softmax = torch.exp(output)
			idx = target_keys[argmax(output)[1]]
			self:updateAccuracy(idx, target)
		end
	end
	print("got -|" .. total_loss / self.counter .. "|- RMSE on -|" .. self.counter .."|- validation samples")
	print(self:statsToString())
	return total_loss / self.counter
end

function ActivityImputer:validateConfidence( data )
local total_loss = 0
	self:zeroStats()
	local obs = self.see_future and 1 or 2

	for i=self.range + 1, data:size(1) - self.range do
		local sample = data[{{i - self.range, i + self.range}}]:clone()
		local target = sample:clone()[obs * self.range + 1][2]
		sample[obs * self.range + 1] = 0
		local input = self:format(sample)
		if target ~= 0 and sample:ne(0):sum() > self.min_obs then
			target = target_keys[target]
			local output = self:forward(input)
			local softmax = torch.exp(output)
			total_loss = total_loss + self.criterion:forward(output, target)
			idx = target_keys[argmax(softmax:cmul(self.criterion.weights))[1]]
			self:updateAccuracy(idx, target)
		end
	end
	print("got -|" .. total_loss / self.counter .. "|- RMSE on -|" .. self.counter .."|- validation samples")
	print(self:statsToString())
	return total_loss / self.counter
end

function ActivityImputer:batchProb( data )
	
	-- returns a tensor of log probabilities for each class for each sample 
	local out = torch.Tensor(data:size(1), self.act_types):fill(0)
	local start_buf = self.see_future and self.range + 1 or 2 * self.range + 1
	local end_buf = self.see_future and self.range or 0

	for i=start_buf, data:size(1) - end_buf do
		local sample = nil
		if self.see_future == true then
			sample = data[{{i - self.range, i + self.range}}]:clone()
		else
			sample = data[{{i - 2 * self.range, i}}]
		end
		local input = self:format(sample)
		if input:sum() > 0 then
			local weighted = torch.exp(self:forward(input)):cmul(self.criterion.weights)
			out[i] =  weighted / weighted:sum()
		else
			out[i] = self:forward(input):fill(0)
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

function ActivityImputer:train( data, n, epoch_size )

	-- performs one epoch of training 

	batch_size = epoch_size or 1
	collectgarbage()
	self.total_mse = 0
	self.counter = 0
	local obs = self.see_future and 1 or 2

	local shuffle_idxs = torch.randperm(data:size(1) - 2*self.range):add(self.range)
	for idx=1, math.floor(shuffle_idxs:size(1) * batch_size) do

		local i = shuffle_idxs[idx] 
		local sample = data[{{i - self.range, i + self.range}}]:clone()
		local target = sample:clone()[obs * self.range + 1][2]
		sample[obs * self.range + 1] = 0
		sample = augment_time(sample, 4)
		local input = self:format(sample)
		if target ~= 0 and sample:ne(0):sum() > self.min_obs then
			target = target_keys[target]
			output = self:forward(input)			
			local mseloss = self.criterion:forward(output, target)
			self:updateTrainStats(mseloss)
			self:zeroGradParameters()
			local gradient = self.criterion:backward(output, target)
			gradient = self:clipGradTensor(gradient, self.maxgrad)
			self:backward(input, gradient)
			local current_learning_rate = self.learning_rate  * self.learning_rate_decay
			self:updateParameters(current_learning_rate)
		end
	end
	print("epoch " .. n .. " got " .. self.total_mse / self.counter .. " MSE on " .. self.counter .." training samples")
end

function test()
	local min_loss = 1000
	local training_data, valid_data, test_data = train_test_split(arg[1])
	print(training_data:size())
	print(valid_data:size())
	training_data = training_data[{{},{training_data:size(2) -1, training_data:size(2)}}]
	valid_data = valid_data[{{},{valid_data:size(2) - 1, valid_data:size(2)}}]
	local s = training_data:ne(0):sum()
	print(s)
	local num_acts = 8
	local weights = torch.Tensor(num_acts)

	for i=1,weights:size(1) do
		weights[i] = (1 - (training_data:eq(i):sum() / s))
	end
	weights = weights / weights:sum()
	print(weights)
	local net = nn.ActivityImputer(num_acts,weights)
	print(net)

	print(training_data:size(1), valid_data:size(1))
	for i=1,100 do
		net:train(training_data,i)
		net:updateVisuals(valid_data)

		local loss = net:valid(valid_data)
		net:nearestGuessEval(valid_data)
		net:modeEval(valid_data)

		print(net.actual_totals)
		if loss < min_loss then
			min_loss = loss
			torch.save("activity_kernel.t7", net)
			print("network saved")
			-- look at one week
		end
	end
end

function apply_model( model_fn, data_fn )
	local net = torch.load(model_fn)
	local training_data, valid_data, test_data = train_test_split()
	local data = training_data:cat(valid_data, 1):cat(test_data, 1)
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

