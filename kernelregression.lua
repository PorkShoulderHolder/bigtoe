require 'cunn'
require 'nn'
require 'torch'
require 'cutorch'
require 'gnuplot'
torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(2)
local opt = lapp[[
	--gpuid           (default 1)
	--augment_time    (default 1)
	--flag_limit_gradient	 (default 1)
	--flag_control_bias (default 1)
]]
augment_time = opt.augment_time
flag_limit_gradient = opt.flag_limit_gradient
flag_control_bias = opt.flag_control_bias
cutorch.setDevice(opt.gpuid)
print(opt)

function  init()
	learningRate = 0.01
	sigma2 = 10
	basis = torch.range(-30,30)
	normal_kernel = torch.exp(torch.div(torch.abs(basis), -1*sigma2)):fill(1)
	halfkwidth = math.floor(normal_kernel:size(1)/2)
	max_grad = 10
	max_bias = 10
	gaussian_noise_var_x = 0.05
	gaussian_noise_var_t = 1
	learningRateDecay = 0.01

	x = dofile('readbinary.lua')

	labels_all = {}
	labelsFile  = io.open('../../../../baseline5mil/config/loinc_file.top1000.withLabels')
	local line = labelsFile:read("*l")
	while (line ~= nil) do
	   table.insert(labels_all,line)
	   line = labelsFile:read("*l")
	end
	labelsFile:close()
end

function setup_network(labix, countX)
	print(labels_all[labix])
	x1 = x[{{labix},{1,countX},{}}]:cuda()

	big_model = nn.Sequential()

	conv_ratio = nn.ParallelTable()
	conv_layer_top = nn.SpatialConvolutionMM(1,1,halfkwidth*2+1,1)
	conv_layer_top.weight = normal_kernel:viewAs(conv_layer_top.weight)
	conv_layer_top.bias = torch.Tensor({0}):viewAs(conv_layer_top.bias)
	conv_layer_clone_bott = conv_layer_top:clone('weight','bias')	
	conv_ratio:add(conv_layer_top)
	conv_ratio:add(conv_layer_clone_bott)

	big_model:add(conv_ratio)
	big_model:add(nn.CDivTable())

	big_model = big_model:cuda()
	conv_layer_clone_bott:share(conv_layer_top,'weight','bias')

	criterion = nn.MSECriterion():cuda()
	dmsedf_table = {}
	mseloss_table = {}
end

function pad(input, padding)
	res = torch.Tensor(input:size(1),input:size(2),input:size(3),input:size(4)+padding+padding):fill(0):cuda()
	res[{{},{},{},{1+padding, input:size(4)+padding}}] = input
	return res
end

function normalize(input)
	local inputnnx = input:ne(0)
	local mean = input:sum()/inputnnx:sum()
	local std = torch.sqrt( torch.pow(input,2):sum()/inputnnx:sum() - mean*mean )

	input = input - mean
	input = torch.cmul(input, inputnnx)

	if std > 0 then
		input = input/std
	end
	return input:clone(), inputnnx:clone(), mean, std	
end

function normalize_target(target, mean, std)
	target = target - mean
	if std > 0 then
		target = target/std
	end
	return torch.CudaTensor({target})
end

function regress()
	total_mse = 0
	total_mse_counter = 0
	for i= 1,x1:size(2) do
		if x1[{{1},{i},{}}]:gt(0):sum() > 2 then
			local input = pad(x1[{{1},{i},{}}]:view(1,1,1,109), halfkwidth):clone()			
			input, inputnnx, mean, std = normalize(input)

			local output = big_model:forward({input, inputnnx})
			local results = (output * std) + mean
			local target =  x1[{{1},{i},{}}]:cuda()

			for t =1, x1:size(3) do
				if inputnnx[1][1][1][t+halfkwidth] ~= 0 then
					total_mse = total_mse + (results:squeeze()[t]-target:squeeze()[t]) * (results:squeeze()[t]-target:squeeze()[t])
					total_mse_counter = total_mse_counter + 1
				end
			end				
		end
	end
	print('regress:')
	print(math.sqrt(total_mse/total_mse_counter))
end

function limit_value(inputx, max_value)
	if inputx > max_value then
		inputx = max_value
	elseif inputx < -1*max_value then
		inputx = -1*max_value
	end
	return inputx
end

function augment_input(input, t)
	local nnx = input:ne(0)
	local gaussian_noise_vector = (torch.randn(input:size()):cuda() * gaussian_noise_var_x)
	local newinput = input + torch.cmul(nnx, gaussian_noise_vector)
	nnx = nnx:squeeze()
	if augment_time == 1 then
		for tix = halfkwidth+1, input:size(4)-halfkwidth do
			if nnx[tix] == 1 and tix ~= t then
				local jump = math.floor((torch.randn(1) * gaussian_noise_var_t):squeeze())
				if tix+jump > 0 and tix+jump < input:size(4)+1 then
					local tmp_input = newinput[1][1][1][tix+jump]
					newinput[{{1},{1},{1},{tix+jump}}] = input[{{},{},{},{tix}}]:squeeze()
					newinput[{{1},{1},{1},{tix}}]= tmp_input
				end
			end			
		end		
	end
	return newinput:clone()
end

function train(maxEpoch)
	big_model:training()
	for epoch = 1,maxEpoch do		
		collectgarbage()
		total_mse = 0
		total_mse_counter = 0
		gnuplot.figure(1)
		gnuplot.plot({'top_convnet',conv_layer_top.weight:float():squeeze(), '-'},{'bottom_convnet',conv_layer_clone_bott.weight:float():squeeze(), '-'})
		print('bias');print(conv_layer_top.bias);print(conv_layer_clone_bott.bias)
		print ('epoch'..epoch)
		shuffled_ix = torch.randperm(x1:size(2))
		shuffled_time = torch.randperm(x1:size(3))
		regress()		

		for ox = 1, x1:size(3)*x1:size(2) - 1 do
			tx = math.fmod(ox,109); if tx == 0 then; tx = 109; end;
			ix = math.floor(ox/109) + 1
			t = shuffled_time[tx]			
			i = shuffled_ix[ix]			
			if x1[1][i][t] ~= 0 and x1[{{1},{i},{}}]:gt(0):sum() > 2 then
				--create mini-batch

				big_model:zeroGradParameters()	

				padded_t = t + halfkwidth
				local input = pad(x1[{{1},{i},{}}]:view(1,1,1,109), halfkwidth):clone()				
				local target = input[1][1][1][padded_t]
				input[{{1},{1},{1},{padded_t}}]:fill(0)
				input = augment_input(input,padded_t)

				input, inputnnx, mean, std = normalize(input)
				target = normalize_target(target, mean, std)					
				if (std > 10) then
					print(ix .. ' ' ..tx .. 'std'.. std.. 'mean'..mean)
				end
								
				local output = big_model:forward({input, inputnnx})
				local mseloss = criterion:forward(output[{{1},{1},{1},{t}}], target)
				local msegd = criterion:backward(output[{{1},{1},{1},{t}}], target):squeeze()
				if flag_limit_gradient == 1 then
					msegd = limit_value(msegd, max_grad)					
				end
				backward_gradients = output:clone():zero()
				backward_gradients[{{1},{1},{1},{t}}]:fill(msegd)
				
				big_model:backward({input, inputnnx}, backward_gradients)
				current_learning_rate = learningRate / (1 + epoch * learningRateDecay)          
				big_model:updateParameters(current_learning_rate)
				if flag_control_bias == 1 then
					local tmp = limit_value(conv_layer_top.bias:squeeze(), max_bias)
					conv_layer_top.bias = torch.CudaTensor({tmp}):viewAs(conv_layer_top.bias)
					conv_layer_clone_bott:share(conv_layer_top,'bias')
				end
				-- gnuplot.figure(1)
				-- gnuplot.plot({'top_convnet',conv_layer_top.weight:float():squeeze(), '-'},{'bottom_convnet',conv_layer_clone_bott.weight:float():squeeze(), '-'})		
				total_mse = mseloss + total_mse
				total_mse_counter = 1 + total_mse_counter
			end		
		end
		print(total_mse/total_mse_counter)		
	end
end

-- for i = 1, 100 do	
-- 	setup_network(i)
-- 	regress()
-- end

init()
setup_network(4, 5000)
train(100)
