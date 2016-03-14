require 'cunn'
require 'torch'
require 'nn'
require 'gnuplot'
require 'kernelutil'
require 'visualizer' 

local ActivityImputer, parent = torch.class('nn.ActivityImputer', 'nn.Sequential')

function ActivityImputer:__init(act_types)
	-- body
   parent.__init(self)

	self.act_types = act_types or 6
	self.range = range or 120
	self.min_obs = 2
	self.maxgrad = 1
	self.learning_rate = 1
	self.hide_exogenous = false
	self.learning_rate_decay = 0.01
	
	self:zeroStats()

	self.averaging_pd = 11
	self.criterion = nn.ClassNLLCriterion()
	self.kernel = nn.Linear(self.act_types* ((2*self.range + 1) - self.averaging_pd),self.act_types)
	self:add(nn.SpatialAveragePooling((2*self.range + 1) - self.averaging_pd, self.act_types, 1, 1))
	self:add(nn.Reshape(self.act_types* ((2*self.range + 1) - self.averaging_pd)))
	self:add(self.kernel)
	self:add(nn.LogSoftMax())
end

function ActivityImputer:format(sample)
	--
	-- expects a tensor [range] x 1 of labels
	--
	local out = torch.Tensor(self.act_types,2*self.range + 1)
	for i=1,self.act_types do
		out[{i,{}}] = sample:eq(i)
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
	gnuplot.splot(self.kernel.weight[{2,{}}]:view(self.act_types,2*self.range + 1))
	gnuplot.raw('set multiplot layout 2,1')
	draw_onehot(data[{{2500, 3111}}], self)
	draw_onehot_nll(data[{{2500, 3111}}], self)
	gnuplot.raw('unset multiplot')
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
		out = out .. k .. " - " .. v/self.actual_totals[k] .. ", " 
	end
	return out:sub(1,-3)
end

function ActivityImputer:nearestGuessEval( data )

	-- method used for benchmarking

	self:zeroStats()
	for i=self.range + 1, data:size(1) - self.range do
		local sample = data[{{i - self.range, i + self.range}}]:clone()
		local target = sample:clone()[self.range + 1]
		sample[self.range + 1] = 0
		if target ~= 0 and sample:ne(0):sum() > self.min_obs then
			local nid = nearestIndex(sample, self.range + 1)
			local guess = sample[nid]
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
		local target = sample:clone()[self.range + 1]
		sample[self.range + 1] = 0
		if target ~= 0 and sample:ne(0):sum() > self.min_obs then
			local counts = torch.Tensor({sample:eq(1):sum(),sample:eq(2):sum(),sample:eq(3):sum(),
										sample:eq(4):sum(),sample:eq(5):sum(),sample:eq(6):sum()})
			self:updateAccuracy(argmax(counts)[1], target)
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
	for i=self.range + 1, data:size(1) - self.range do
		local sample = data[{{i - self.range, i + self.range}}]:clone()
		local target = sample:clone()[self.range + 1]
		sample[self.range + 1] = 0
		local input = self:format(sample)
		if target ~= 0 and sample:ne(0):sum() > self.min_obs then
			local output = self:forward(input)
			total_loss = total_loss + self.criterion:forward(output, target)
			local softmax = torch.exp(output)
			idx = argmax(softmax)[1]
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
	for i=self.range + 1, data:size(1) - self.range do
		local sample = data[{{i - self.range, i + self.range}}]:clone()
		local input = self:format(sample)
		out[i] = torch.exp(self:forward(input))
	end
	return out
end

function ActivityImputer:train( data, n, epoch_size )

	-- performs one epoch of training 

	batch_size = epoch_size or 1
	collectgarbage()
	self.total_mse = 0
	self.counter = 0
	local shuffle_idxs = torch.randperm(data:size(1) - 2*self.range):add(self.range)
	for idx=1, math.floor(shuffle_idxs:size(1) * batch_size) do

		local i = shuffle_idxs[idx] 
		local sample = data[{{i - self.range, i + self.range}}]:clone()
		local target = sample:clone()[self.range + 1]
		sample[self.range + 1] = 0
		--sample = augment_time(sample)

		local input = self:format(sample)
		if target ~= 0 and sample:ne(0):sum() > self.min_obs then
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
args = { ... }

function test()
	local net = nn.ActivityImputer()
	print(net)
	local min_loss = 1000
	local training_data, valid_data = train_test_split(args[1])
	training_data = training_data[{{},3}]
	valid_data = valid_data[{{},3}]
	print(training_data:size(1), valid_data:size(1))
	for i=1,100 do
		net:train(training_data,i)
		local loss = net:valid(valid_data)
		net:nearestGuessEval(valid_data)
		net:modeEval(valid_data)
		net:updateVisuals(valid_data)

		print(net.actual_totals)
		if loss < min_loss then
			min_loss = loss
			torch.save("activity_kernel.t7", net)
			print("network saved")
			-- look at one week
		end
	end

end



test()



