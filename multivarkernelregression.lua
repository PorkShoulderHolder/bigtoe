require 'cunn'
require 'nn'
require 'torch'
require 'cutorch'
require 'gnuplot'
require 'paths'

torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(2)
local opt = lapp[[
	--gpuid           (default 1)
	--augment_time    (default 1)
	--flag_limit_gradient	 (default 1)
	--flag_control_bias (default 1)
	--labix 		(default 1)
]]

augment_time = opt.augment_time
flag_limit_gradient = opt.flag_limit_gradient
flag_control_bias = opt.flag_control_bias
cutorch.setDevice(opt.gpuid)
torch.setnumthreads(1)
print(opt)

function  init()
	learningRate = 0.01
	sigma2 = 10
	basis = torch.range(-10,10)
	normal_kernel = torch.exp(torch.div(torch.abs(basis), -1*sigma2))
	halfkwidth = math.floor(normal_kernel:size(1)/2)
	max_grad = 10
	max_bias = 10
	gaussian_noise_var_x = 0.05
	gaussian_noise_var_t = 1
	learningRateDecay = 0.01

	x = dofile('readbinary.lua')
	labcounts = x:size(1)
	timecounts = x:size(3)
	peoplecounts = x:size(2)

	labels_all = {}
	labelsFile  = io.open('../../../../baseline5mil/config/loinc_file.top1000.withLabels')
	local line = labelsFile:read("*l")
	while (line ~= nil) do
	   table.insert(labels_all,line)
	   line = labelsFile:read("*l")
	end
	labelsFile:close()
end

function covariance(x1)
	local xmax, dummy = x1:max(3)
	xmax = xmax:squeeze()
	local xmean = xmax:mean(2):squeeze()
	xmean = xmean:view(xmean:size(1),1)
	xmeanzero = xmax - xmean:expand(xmax:size(1),xmax:size(2))
	return torch.mm(xmeanzero,xmeanzero:t())	
end

function setup_network(labix, countX)
	print(labels_all[labix])
	x1 = x[{{},{1,countX},{}}]:cuda()
	x1valid = x[{{},{countX+1,peoplecounts},{}}]:cuda()
	covmatrix = covariance(x)
	-- gnuplot.figure(10)
	-- gnuplot.imagesc(covmatrix)

	cov_row = covmatrix[{{},{labix}}] / covmatrix[labix][labix]
	kernel_matrix_init = torch.mm(cov_row,normal_kernel:view(1,normal_kernel:size(1))):fill(0.1)
	-- gnuplot.figure(9)
	-- gnuplot.imagesc(kernel_matrix_init)

	big_model = nn.Sequential()

	conv_ratio = nn.ParallelTable()
	conv_layer_top = nn.SpatialConvolutionMM(1,1,halfkwidth*2+1,covmatrix:size(1),1,1,halfkwidth,0)
	conv_layer_top.weight = kernel_matrix_init:viewAs(conv_layer_top.weight)
	conv_layer_top.bias = torch.Tensor({0}):viewAs(conv_layer_top.bias)
	conv_layer_clone_bott = conv_layer_top:clone('weight','bias')	
	conv_ratio:add(conv_layer_top)
	conv_ratio:add(conv_layer_clone_bott)

	big_model:add(conv_ratio)
	big_model:add(nn.CDivTable())

	big_model = big_model:cuda()
	conv_layer_clone_bott:share(conv_layer_top,'weight','bias')

	shift_network = nn.SpatialConvolutionMM(1,1,halfkwidth*2+1,covmatrix:size(1),1,1,halfkwidth,0):cuda()

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
	local mean = torch.cdiv(input:sum(4),inputnnx:sum(4)):squeeze()
	mean[inputnnx:sum(4):squeeze():abs():eq(0)] = 0.0	

	local std = torch.cdiv(torch.pow(input,2):sum(4), inputnnx:sum(4)):squeeze() - torch.cmul(mean,mean)
	std[inputnnx:sum(4):squeeze():eq(0)] = 0.0
	std[std:lt(0)]=0.0
	std = torch.sqrt(std)
	
	std = std:view(std:size(1),1)
	mean = mean:view(mean:size(1),1)

	input = input - mean:expand(mean:size(1),input:size(4))
	input = torch.cmul(input, inputnnx)
	stdtmp = std:clone()
	stdtmp[stdtmp:eq(0)] = 1.0
	input = torch.cdiv(input, stdtmp:expand(stdtmp:size(1),input:size(4)))	
	return input:clone(), inputnnx:clone(), mean, std
end

function normalize_target(target, mean, std)
	target = target - mean
	if std > 0 then
		target = target/std
	end
	return torch.CudaTensor({target})
end

function regress(labix)
	total_mse = 0
	total_mse_counter = 0
	for i= 1,x1valid:size(2) do
		if x1valid[{{labix},{i},{}}]:gt(0):sum() > 2 then
			local input = x1valid[{{},{i},{}}]:clone():view(1,1,labcounts,timecounts):cuda()
			input, inputnnx, mean, std = normalize(input)			

			local output = big_model:forward({input, inputnnx})
			local results = (output * std[labix]:squeeze()) + mean[labix]:squeeze()
			local target = x1valid[{{labix},{i},{}}]:cuda()			

			for t =1, timecounts do
				if inputnnx[1][1][labix][t] ~= 0 then
					total_mse = total_mse + (results:squeeze()[t] - target:squeeze()[t]) * (results:squeeze()[t] - target:squeeze()[t])
					total_mse_counter = total_mse_counter + 1
				end
			end				
		end
	end
	print('regress:')
	print(math.sqrt(total_mse/total_mse_counter))
	return(math.sqrt(total_mse/total_mse_counter))
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

	if augment_time == 1 then		
		for labixx = 1, input:size(3) do
			nnxlab = nnx[{{},{},{labixx},{}}]:ne(0):squeeze() --1x169
			for tix = 1, input:size(4) do			
				if nnxlab[tix] == 1 and tix ~= t then
					local jump = math.floor((torch.randn(1) * gaussian_noise_var_t):squeeze())
					if tix+jump > 1 and tix+jump < input:size(4) then
						local tmp_input = newinput[1][1][labixx][tix+jump]
						newinput[{{1},{1},{labixx},{tix+jump}}] = input[{{},{},{labixx},{tix}}]:squeeze()
						newinput[{{1},{1},{labixx},{tix}}]= tmp_input
					end
				end			
			end		
		end
	end
	return newinput:clone()
end

function train(maxEpoch, labix)
	big_model:training()
	for epoch = 1,maxEpoch do		
		collectgarbage()
		total_mse = 0
		total_mse_counter = 0
		print ('epoch'..epoch)
		print('bias'); print(conv_layer_top.bias); print(conv_layer_clone_bott.bias)		
		shuffled_ix = torch.randperm(x1:size(2))
		shuffled_time = torch.randperm(x1:size(3))
		validScore = regress(labix)
		for ox = 1, x1:size(3)*x1:size(2) - 1 do
			tx = math.fmod(ox,timecounts); if tx == 0 then; tx = timecounts; end;
			ix = math.floor(ox/timecounts) + 1
			t = shuffled_time[tx]		
			i = shuffled_ix[ix]			
			if x1[labix][i][t] ~= 0 and x1[{{labix},{i},{}}]:gt(0):sum() > 2 then
				big_model:zeroGradParameters()	

				local input = x1[{{},{i},{}}]:clone():view(1,1,labcounts,timecounts):cuda()
				local target = input[1][1][labix][t]
				input[{{1},{1},{labix},{t}}]:fill(0)
				input = augment_input(input,t)

				input, inputnnx, mean, std = normalize(input)
				target = normalize_target(target, mean[labix][1], std[labix][1])				
		
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
				total_mse = mseloss + total_mse
				total_mse_counter = 1 + total_mse_counter			
			end		
		end
		gnuplot.figure(1)
		gnuplot.splot(conv_layer_top.weight:float():view(labcounts,normal_kernel:size(1)):clone())--,'color'		
		-- gnuplot.figure(2)
		-- gnuplot.plot(conv_layer_top.weight:float():view(labcounts,normal_kernel:size(1)):clone():abs():mean(1):squeeze(),'-')
		-- gnuplot.figure(3)
		-- gnuplot.plot(conv_layer_top.weight:float():view(labcounts,normal_kernel:size(1)):clone():mean(2):squeeze(),'-')
		print(total_mse/total_mse_counter)
		local filename = paths.concat('networks/lab'.. labix ..'_epoch'..epoch ..'.net')
		os.execute('mkdir -p ' .. sys.dirname(filename))
		if paths.filep(filename) then
		  os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
		end
		print('Saving network to '..filename)
		torch.save(filename, big_model)	
	end
end

init()
setup_network(opt.labix, 5000)
train(100,opt.labix)
