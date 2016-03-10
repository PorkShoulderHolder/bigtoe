require 'torch'
require 'nn'
require 'gnuplot'


local ActivityImputer, parent = torch.class('nn.ActivityImputer', 'nn.Sequential')

function ActivityImputer:__init(act_types)
	-- body
   parent.__init(self)

	self.act_types = act_types or 6
	self.range = range or 30
	self.min_obs = 2
	self.maxgrad = 1
	self.learning_rate = 1.8
	self.hide_exogenous = false
	self.learning_rate_decay = 0.01

	self.criterion = nn.ClassNLLCriterion()
	self.kernel = nn.Linear(self.act_types* (2*self.range + 1),self.act_types)
	self:add(nn.Reshape(self.act_types* (2*self.range + 1)))
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

function ActivityImputer:updateVisuals( )
	-- body
	gnuplot.figure(1)


	--gnuplot.imagesc(self.kernel.weight[{1,{}}]:view(self.act_types,2*self.range + 1),'color')
	gnuplot.raw('set multiplot layout 6,1')
	gnuplot.imagesc(self.kernel.weight[{1,{}}]:view(self.act_types,2*self.range + 1):clone(),'color')
	gnuplot.imagesc(self.kernel.weight[{2,{}}]:view(self.act_types,2*self.range + 1):clone(),'color')
	gnuplot.imagesc(self.kernel.weight[{3,{}}]:view(self.act_types,2*self.range + 1):clone(),'color')
	gnuplot.imagesc(self.kernel.weight[{4,{}}]:view(self.act_types,2*self.range + 1):clone(),'color')
	gnuplot.imagesc(self.kernel.weight[{5,{}}]:view(self.act_types,2*self.range + 1):clone(),'color')
	gnuplot.imagesc(self.kernel.weight[{6,{}}]:view(self.act_types,2*self.range + 1):clone(),'color')
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

function ActivityImputer:train( data, epoch )
	collectgarbage()
	self.total_mse = 0
	self.counter = 0
	local shuffle_idxs = torch.randperm(data:size(1) - 2*self.range):add(self.range)
	for idx=1, shuffle_idxs:size(1)/10 do
		
	  --  if(idx % 500 == 1) then self:updateVisuals() end
		local i = shuffle_idxs[idx] 
		local sample = data[{{i - self.range, i + self.range}}]:clone()
		local target = sample:clone()[self.range + 1]
		sample[self.range + 1] = 0
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
	self:updateVisuals()
	print("epoch " .. epoch .. " got " .. self.total_mse / self.counter .. " MSE on " .. self.counter .." training samples")
end
args = { ... }

local net = nn.ActivityImputer()
x = assert(loadfile('readjson.lua'))(args[1])
print(net)
local data = x[{{},3}]
print(data:size())

for i=1,1000 do
	net:train(data,i)
end




