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
	self.min_obs = 6
	self.maxgrad = 1
	self.bg_spacing = 20
	self.learning_rate = 2
	self.learning_rate_decay = 0.001
	self.see_future = false
	self.include_exog = false
	self:zeroStats()
	self.denom = false
	self.prediction_horizon_bg = 5 -- # of bg measurements to mask; 5 = 30 mins, 11 = 1 hr 
	self.prediction_horizon_onehot = self.prediction_horizon_bg * self.bg_spacing
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

	self.bottom_scalar = self.top_scalar:clone('weight','bias')
	self.bottom_scalar:share(self.top_scalar,'weight', 'bias')

	self.top_onehot = nn.SparseLinear(self.onehot_count * (2 * self.bg_spacing * self.range + 2), 1)
	self.top_onehot.weight:fill(0.001)
	self.numer_table:add(self.top_scalar)
	self.numer_table:add(self.bottom_scalar)

	self.numerator:add(self.numer_table)
	self.numerator:add(CDivTable_robust())
	self.container:add(self.numerator)

	self.denominator = nn.Sequential()
	--self.denom_table = nn.ParallelTable()
	
	--self.denom_table:add(self.top_onehot)
	--self.denom_table:add(nn.Identity())
	self.denominator:add(self.top_onehot)
	--self.denominator:add(CDivTable_robust())
	self.container:add(self.denominator)

	self:add(self.container)
	self:add(nn.CAddTable())
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
			out[i] = torch.Tensor({self.onehot_count * j + cluster, 1})
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

function ScalarOneHot:updateVisuals( slices_of_interest  )

	-- body
	gnuplot.figure(1)
	local top = self.top_scalar.weight[{1,{}}]:view(2*self.range):squeeze()
	local bottom = self.bottom_scalar.weight[{1,{}}]:view(2*self.range)
	local ws = self.top_onehot.weight:squeeze()
	local avgs = torch.Tensor(2 * self.range * self.bg_spacing + 1):fill(0)
	
	local sparse_data = ws:view(2*self.range * self.bg_spacing + 2, self.onehot_count)

	local avgs1 = torch.mean(sparse_data,1)
	local avgs2 = torch.mean(sparse_data,2)

	--gnuplot.plot({'thai', sparse_data[{{},43}], '-'}) --, {'15',sparse_data[{{},15}], '-'})
	--gnuplot.plot( {'1',sparse_data[{{},1}], '-'}, {'12',sparse_data[{{},12}], '-'}, {'10', sparse_data[{{},10}], '-'}, {'44',sparse_data[{{},44}], '-'} )
	--

	local slices = 0
   if slices_of_interest then
      slices = table.getn(slices_of_interest)
   end
   gnuplot.raw('set multiplot layout ' .. 2 + slices ..  ',1')
	gnuplot.plot({'avg loc importance', avgs2, '-'}) --, {'15',sparse_data[{{},15}], '-'})
	gnuplot.plot({'bg_weights', top, '-'}) --, {'15',sparse_data[{{},15}], '-'})
   if slices > 0 then
      for i,j in pairs(slices_of_interest) do
         gnuplot.plot({'individual cluster importance', sparse_data[{{},j}], '-'})
      end
   end
	--draw_onehot(data[{{2500, 12000}}], self)
	--draw_onehot_nll(data[{{2500, 12000}}], self)
	gnuplot.raw('unset multiplot')
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

function ScalarOneHot:prepare_input( data_scalar, data_onehot, i )
	local obs = self.see_future and 1 or 2
	local onehot_sample = data_onehot[{{i - 20 * self.range, i + 20 *self.range}}]:clone()
	local scalar_sample = data_scalar[{{i - 20 * self.range, i + 20 *self.range}}]:clone()
	local target = scalar_sample:clone()[20 * obs * self.range + 1]
	
	--scalar_sample = augment_time(scalar_sample, 1)
	scalar_sample = self:format_scalar(scalar_sample)

	scalar_sample[{{obs * self.range - self.prediction_horizon_bg, obs * self.range}}] = 0
	onehot_sample[{{obs * self.range * self.bg_spacing + (1 - self.prediction_horizon_onehot), obs * self.range * self.bg_spacing + 1}}] = 0
	--onehot_sample[{{1, self.prediction_horizon_onehot}}] = 0
	return scalar_sample, onehot_sample, target
end

function ScalarOneHot:getClusterWeights( n )
	local ws = self.top_onehot.weight:squeeze()
	local sparse_data = ws:view(2*self.range * self.bg_spacing + 2, self.onehot_count)
	return sparse_data[{{}, n}]
end

function ScalarOneHot:setClusterWeights( n , scalar_val )
	self.top_onehot.weight:view(2*self.range * self.bg_spacing + 2, self.onehot_count)[{{}, n}]:fill(scalar_val)
end

function ScalarOneHot:train( data_scalar, data_onehot, n, epoch_size )

	-- performs one epoch of training 
	
	-- we set params that will never be touched to zero (as opposed to rondom) to improve generalization

	batch_size = epoch_size or 1
	collectgarbage()
	self.total_mse = 0
	self.counter = 0
	local obs = self.see_future and 1 or 2
	local locscnt = 0
	local nonzeros = data_scalar:nonzero()
	local obs_counts = {}
	for i=1,self.onehot_count do
		obs_counts[i] = 0
	end
	nonzeros = nonzeros[nonzeros:gt(2 * self.range * 20)]

	local shuffle_idxs = torch.randperm(nonzeros:size(1))
	print(math.floor(shuffle_idxs:size(1) * batch_size))
	for idx=1, math.floor(shuffle_idxs:size(1) * batch_size) do
		local i = nonzeros[shuffle_idxs[idx]] - (self.range * 20)
		
		local scalar_sample, onehot_sample, target = self:prepare_input( data_scalar, data_onehot , i)
		if target ~= 0 and scalar_sample:ne(0):sum() > self.min_obs then
			local scalar_input, mean, std = self:normalize(scalar_sample)
			target = (target - mean) / std


			if(self.include_exog ~= true) then
				onehot_sample:fill(0)
			end
			for ig=1, self.onehot_count do
				obs_counts[ig] = obs_counts[ig] + onehot_sample:eq(ig):sum() 
			end


			onehot_sample = augment_time(onehot_sample, self.bg_spacing)
		
			local onehot_input = self:format(onehot_sample)

			local onehot_denom = torch.Tensor({1})
			if onehot_sample:clone():ne(0):sum() > 0 and self.denom == true then 
				onehot_denom = torch.Tensor({onehot_sample:clone():ne(0):sum()})
			end

			local input_table = {
									{scalar_input:double(), scalar_sample:clone():ne(0):double()}, 
									onehot_input:double()
								}

			local output = self:forward(input_table)
			target = torch.Tensor({target})
			local mseloss = self.criterion:forward(output, target)
			self:updateTrainStats(math.sqrt(mseloss) * std)
			self:zeroGradParameters()

			local gradient = self.criterion:backward(output, target)
			gradient = self:clipGradTensor(gradient, self.maxgrad)
			self:backward(input_table, gradient)
			local current_learning_rate = self.learning_rate  * self.learning_rate_decay
			self:updateParameters(current_learning_rate)
		end
	end
	for k,v in pairs(obs_counts) do
		if v == 0 then
			self:setClusterWeights(k, 0)
		end
	end
	--self:updateVisuals()
	print("epoch " .. n .. " got " .. self.total_mse / self.counter .. " MSE on " .. self.counter .." training samples and " .. locscnt)
end

function ScalarOneHot:valid( data_scalar, data_onehot )
	
	-- method used for validation

	lbatch_size = epoch_size or 1
	collectgarbage()
	self.total_mse = 0
	self.counter = 0
	local obs = self.see_future and 1 or 2
	local locscnt = 0
	local nonzeros = data_scalar:nonzero()

	nonzeros = nonzeros[nonzeros:gt(2 * self.range * 20)]

	for idx=1, nonzeros:size(1) do
		local i = nonzeros[idx] - (self.range * 20)

		local scalar_sample, onehot_sample, target = self:prepare_input( data_scalar, data_onehot , i)

		if target ~= 0 and scalar_sample:ne(0):sum() > self.min_obs then
			local scalar_input, mean, std = self:normalize(scalar_sample)
			target = (target - mean) / std

			if(self.include_exog ~= true) then
				onehot_sample:fill(0)
			end


			local onehot_denom = torch.Tensor({1})
			if onehot_sample:clone():ne(0):sum() > 0 and self.denom == true then 
				onehot_denom = torch.Tensor({onehot_sample:clone():ne(0):sum()})
			end


			local onehot_input = self:format(onehot_sample)
			local input_table = {
									{scalar_input:double(), scalar_sample:clone():ne(0):double()  },
									onehot_input:double()
								}
			locscnt = (onehot_sample:eq(-1):sum() > 0) and locscnt + 1 or locscnt
			local output = self:forward(input_table)
			target = torch.Tensor({target})
			local mseloss = self.criterion:forward(output, target)
			self:updateTrainStats(math.sqrt(mseloss) * std)
		end
	end
	print("got -|" .. self.total_mse / self.counter .. "|- MSE on -|" .. self.counter .."|- validation samples")
	--print(self:statsToString())
	return self.total_mse / self.counter
end


function test()
	
	local net = nn.ScalarOneHot(1,58)
	print("horizon: " .. (net.prediction_horizon_bg + 1) * 5 .. " minutes")
	if net.include_exog then print("with exog") else print("without exog") end
	print(net)

	local data = loadjson_asarray( arg[1] )
	print(data:size())
	local td, vv, vd = tts(data) 
	
	print("training_data size " .. td:size(1))
	print("validation_data size " ..  vv:size(1))

	training_data_onehot = td[{{},td:size(2)}]
	valid_data_onehot = vv[{{},vv:size(2)}]

	training_data_scalar_bgs = td[{{},2}]
	valid_data_scalar_bgs = vv[{{},2}]

	local num_acts = 6
	for i=1,1000 do
		net:train(training_data_scalar_bgs, training_data_onehot, i)
		net:valid(valid_data_scalar_bgs, valid_data_onehot)
		torch.save('latest_const.t7', net)
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
elseif arg[2] == 'train' then 
	test()
end

