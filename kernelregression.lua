require 'cunn'
require 'nn'
require 'torch'
require 'cutorch'
require 'gnuplot'
require 'pl'
require 'sys'
require 'kernelutil'

torch.setdefaulttensortype('torch.FloatTensor')
torch.manualSeed(2)
local opt = lapp[[
	--gpuid           (default 1)
	--augment_time    (default 1)
	--flag_limit_gradient	 (default 1)
	--flag_control_bias (default 1)
	--labix 		(default 1)
	--save_train_network_dir (default 'networks_univar_train')
	--save_valid_network_dir (default 'networks_univar_validbest')
	--log_file (default 'log.txt')
	--evaluate_separately (default 1)
]]
augment_time = opt.augment_time
flag_limit_gradient = opt.flag_limit_gradient
flag_control_bias = opt.flag_control_bias
save_train_network_dir = opt.save_train_network_dir
save_valid_network_dir = opt.save_valid_network_dir
cutorch.setDevice(opt.gpuid)
labix = opt.labix
log_file = opt.log_file
evaluate_separately = opt.evaluate_separately

for k, v in pairs(opt) do print(k,v) end

local args = {...}
mode = args[1]
datafilename = args[2]
--labix = args[3]
--save_train_network_dir = args[4]
--save_valid_network_dir = args[5]
--log_file = args[6]


for k, v in pairs(args) do print(k,v) end
print(table.getn(args))

function dump(o)
   if type(o) == 'table' then
      local s = '{ '
      for k,v in pairs(o) do
         if type(k) ~= 'number' then k = '"'..k..'"' end
         s = s .. '['..k..'] = ' .. dump(v) .. ','
      end
      return s .. '}\n '
   else
      return tostring(o)
   end
end

function  init()
	os.execute('mkdir -p ' .. sys.dirname(log_file))
	log_file_open = io.open(log_file, "a")
	log_file_open:write(dump(args))
	log_file_open:write('\n-------- start ------\n')

	learningRate = 0.03
	sigma2 = 10
	range = 30
	basis = torch.range(-1 * range, range)

	normal_kernel = torch.exp(torch.div(torch.abs(basis), -1*sigma2)):fill(1)
	kernel_matrix_init = torch.mm(torch.Tensor(3,1),normal_kernel:view(1,normal_kernel:size(1))):fill(1)

	halfkwidth = math.floor(normal_kernel:size(1)/2)
	max_grad = 0.001
	max_bias = 0.0001
	geo_range = 10
	gaussian_noise_var_x = 0.05
	gaussian_noise_var_t = 1
	learningRateDecay = 0.01
	trainIterations = 100
	peopleCountForTrain = 10000
	peopleCountForValidate = 10000
	batchSize = 100
	batchSizeRegress = 1000

	x = assert(loadfile('readjson.lua'))(datafilename)
	labcounts = x:size(1)
	
	labels_all = {}
--	labelsFile  = io.open('../../../../baseline5mil/config/loinc_file.top1000.withLabels')
--	local line = labelsFile:read("*l")
--	while (line ~= nil) do
--	   table.insert(labels_all,line)
--	   line = labelsFile:read("*l")
--	end
--	labelsFile:close()
end

function setup_network(labix, countX)
	--log_file_open:write(labels_all[labix])
	--log_file_open:write('\n')
	
	--print(labels_all[labix])
	data = x:cuda()
	--x1valid = x[{{labix},{countX+1,peoplecounts},{}}]:cuda()
	big_model = nn.Sequential()

	geo_model = nn.Sequential()
	act_model = nn.Sequential()

	clust_count = 120
	act_types = 6

	map_weights = nn.Linear(clust_count , 1) --(2 * range) + 1)
	act_weights = nn.Linear(act_types , 1)
	


	geo_model:add(map_weights)
	geo_model:add(nn.View(1,1,1,2*range + 1))

	act_model:add(act_weights)
	act_model:add(nn.View(1,1,1,2*range + 1))

	--map_avg = nn.SpatialAveragePooling(geo_range - 1, )

	temporal_ratio = nn.ParallelTable()
	conv_layer_top = nn.SpatialConvolutionMM(1,1,halfkwidth*2+1,3,1,1,halfkwidth,0)
	conv_layer_top.weight = kernel_matrix_init:viewAs(conv_layer_top.weight)
	conv_layer_top.bias = torch.Tensor({0}):viewAs(conv_layer_top.bias)
	conv_layer_clone_bott = conv_layer_top:clone('weight','bias')	

	temporal_ratio:add(conv_layer_top)
	temporal_ratio:add(conv_layer_clone_bott)

	consolidation_table = nn.ParallelTable()
	

	values_model = nn.Sequential()
	values_parallel = nn.ParallelTable()
	
	impulse_model = nn.Sequential()
	impulse_parallel = nn.ParallelTable()

	values_parallel:add(nn.Identity()) -- for bg values (must pass w/ correct view)
	impulse_parallel:add(nn.Identity()) -- for bg impulses (must pass w/ correct view)

	values_parallel:add(geo_model) -- for geo kern
	impulse_parallel:add(nn.Identity()) -- for geo impulses (must pass w/ correct view)
	
	values_parallel:add(act_model) -- for act kern
	impulse_parallel:add(nn.Identity()) -- for act impulses (must pass w/ correct view)
	
	values_model:add(values_parallel)
	values_model:add(nn.JoinTable(2))

	impulse_model:add(impulse_parallel)
	impulse_model:add(nn:JoinTable(2))

	consolidation_table:add(values_model)
	consolidation_table:add(impulse_model)

	big_model:add(consolidation_table)
	big_model:add(temporal_ratio)
	big_model:add(nn.CDivTable())

	print(big_model)
	big_model = big_model:cuda()

	criterion = nn.MSECriterion():cuda()
	dmsedf_table = {}
	mseloss_table = {}
	log_file_open:write('finished building model:')
	log_file_open:write('\n')
end

function load_network(labix, load_network_name, countX)
	print(labels_all[labix])

	log_file_open:write(labels_all[labix])
	log_file_open:write('\n')

	data = x[{{labix},{1,countX},{}}]:cuda()
	big_model = torch.load(load_network_name)
	
	log_file_open:write('finished loading model from ' .. load_network_name)
	log_file_open:write('\n')
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
	return target
end

function regress(data, model)
	total_mse = 0
	total_mse_counter = 0
	
	if (evaluate_separately) then
		local w = model:get(1):get(1).weight		
		local kernel_width =  math.floor(w:size(2)/labcounts)
		w:view(labcounts,kernel_width)[{{labcounts},{math.floor((kernel_width+1)/2)}}]:fill(0)		
	end
	
	batch_input = torch.CudaTensor(batchSizeRegress, 1, 1, timecounts)
	batch_input_nnx = torch.CudaTensor(batchSizeRegress, 1, 1, timecounts)
	batch_target = torch.CudaTensor(batchSizeRegress, 1, timecounts)
	batch_mu_labix = torch.CudaTensor(batchSizeRegress, 1, 1, 1)
	batch_std_labix = torch.CudaTensor(batchSizeRegress, 1, 1, 1)
	bix = 0

	for i= 1,data:size(2) do
		if data[{{1},{i},{}}]:ne(0):sum() > 2 then

			local input = data[{{1},{i},{}}]:view(1,1,1,109):clone()
			input, inputnnx, mean, std = normalize(input)
			local target =  data[{{1},{i},{}}]:cuda()

			bix = bix + 1
			batch_input[{{bix},{1},{},{}}] = input:clone()
			batch_input_nnx[{{bix},{1},{},{}}] = inputnnx:clone()
			batch_target[{{bix},{1},{}}] = target:clone()
			batch_mu_labix[{{bix},{1},{1},{1}}] =  mean
			batch_std_labix[{{bix},{1},{1},{1}}] =  std
			
			if (bix == batchSizeRegress) then
				bix = 0
				local output = model:forward({batch_input, batch_input_nnx})
				local results = torch.cmul(output, batch_std_labix:expand(output:size())) + batch_mu_labix:expand(output:size())				

				local results_nnz = torch.cmul(results, batch_input_nnx):squeeze()
				local targets_nnz = batch_target:squeeze()
				total_mse = total_mse + torch.pow( results_nnz - targets_nnz, 2):sum()
				total_mse_counter = total_mse_counter + batch_input_nnx:sum()
			end
		end
	end
	log_file_open:write('regress finished: ')
	log_file_open:write(math.sqrt(total_mse/total_mse_counter))
	log_file_open:write('\n')
	print('regress:')
	print(math.sqrt(total_mse/total_mse_counter))
	return math.sqrt(total_mse/total_mse_counter)
end

function limit_value(inputx, max_value)
	local pos_mask = inputx:gt(max_value)
	local neg_mask = inputx:lt(-1*max_value)
	inputx[pos_mask] = max_value
	inputx[neg_mask] = -1*max_value
	return inputx
end

function augment_input(input, t)
	local nnx = input:ne(0):cuda()
	local gaussian_noise_vector = (torch.randn(input:size()):cuda() * gaussian_noise_var_x)
	local newinput = input:cuda() + (torch.cmul(nnx, gaussian_noise_vector))
	nnx = nnx:squeeze()
	if augment_time == 1 then
		for tix = 1, nnx:size(1) do
			if nnx[tix] == 1 and tix ~= t then
				local jump = math.floor((torch.randn(1) * gaussian_noise_var_t):squeeze())
				if jump ~= 0 and tix+jump > 0 and tix+jump < (nnx:size(1) + 1) then
					local tmp_input = newinput[1][1][1][tix+jump]
					newinput[{{1},{1},{1},{tix+jump}}] = input[{{},{},{},{tix}}]:squeeze()
					newinput[{{1},{1},{1},{tix}}]= tmp_input
				end
			end			
		end		
	end
	--local rand = torch.rand(s)
	return newinput:clone()
end

function train(maxEpoch)
	big_model:training()
	gnuplot.figure(1)

	for epoch = 1,maxEpoch do		
		collectgarbage()
		total_mse = 0
		total_mse_counter = 0
		print("kern_size" .. conv_layer_top.weight:float():squeeze():size(1))
		--gnuplot.plot({'top_convnet',conv_layer_top.weight:float():squeeze(), '-'},{'bottom_convnet',conv_layer_clone_bott.weight:float():squeeze(), '-'})
		print('bias');print(conv_layer_top.bias);print(conv_layer_clone_bott.bias)
		print ('epoch'..epoch)
		shuffled_ix = torch.randperm(data:size(1))
		--shuffled_time = torch.randperm(data:size(3))
		--regress(data,big_model)		
		--print(data)

		for stort = 1, data:size(1) do
			ox = stort 
			tx = math.fmod(ox,109); if tx == 0 then; tx = 109; end;
			ix = math.floor(ox/109) + 1
			i = shuffled_ix[ox]			
			if i + range > data:size(1) or i - range < 1 then
				goto skip
			end
			if data[i] ~= 0 and data[{{i - range, i + range}}]:gt(0):sum() > range / 2 then
				--create mini-batch
				if(stort % 200 == 0) then
					big_model:updateParameters(current_learning_rate)
					big_model:zeroGradParameters()	
					gnuplot.plot({'top_convnet',conv_layer_top.weight:float():squeeze(), '-'},{'bottom_convnet',conv_layer_clone_bott.weight:float():squeeze(), '-'})		
				end

				local bg_data = data[{{i - range, i + range}, 1}]:clone()
				local bg_input = bg_data:view(1,1,1,2*range + 1):clone()	
				
				local loc_data = build_hist(data[{{i - range, i + range}, 2}], clust_count):clone()
				local loc_input = loc_data:view(1,1, clust_count, 2*range + 1):clone()

				local act_data = build_hist(data[{{i - range, i + range}, 3}], act_types):clone()
				local act_input = act_data:view(1,1, act_types, 2*range + 1):clone()
				
				local bg_target = bg_input:clone()

				-- mask value
				bg_input[{{1},{1},{1},{range}}]:fill(0)
				
				-- perturb inputs				
				bg_input = augment_input(bg_input,range)
				loc_input = augment_input(loc_input,range)
				act_input = augment_input(act_input,range)


				bg_input, bg_inputnnx, bg_mean, bg_std = normalize(bg_input)
				loc_input, loc_inputnnx, loc_mean, loc_std = normalize(loc_input)
				act_input, act_inputnnx, act_mean, act_std = normalize(act_input)
				target = normalize_target(bg_target, bg_mean, bg_std)



				-- if (std > 10) then
				-- 	print(ix .. ' ' ..tx .. ' std'.. std.. ' mean'..mean)
				-- end
			

				local output = big_model:forward({{bg_input, loc_data:cuda(), act_data:cuda()}, 
												 {bg_inputnnx, loc_data:ne(0), act_data:ne(0)}})

				local mseloss = criterion:forward(output, target)

				local msegd = criterion:backward(output, target):squeeze()
				if flag_limit_gradient == 1 then
					msegd = limit_value(msegd, max_grad)					
				end

				
				backward_gradients = output:clone():zero()
				backward_gradients[{{1},{1},{1}}] = msegd

				big_model:backward({input, inputnnx}, backward_gradients)
				current_learning_rate = learningRate / (1 + epoch * learningRateDecay)
				if flag_control_bias == 1 then
					
					local tmp = limit_value(conv_layer_top.bias, max_bias)

					conv_layer_top.bias = tmp
					conv_layer_clone_bott:share(conv_layer_top,'bias')
				end
				--gnuplot.figure(1)
				total_mse = mseloss + total_mse
				total_mse_counter = 1 + total_mse_counter
			end	
			::skip::	
		end

		print('mse' .. total_mse/total_mse_counter)
		log_file_open:write('training epoch mse' .. epoch .. ':')
		log_file_open:write(total_mse/total_mse_counter)
		log_file_open:write('\n')
	
		local filename = paths.concat(save_train_network_dir ..'/lab'.. labix ..'_epoch'..epoch ..'.net')
		os.execute('mkdir -p ' .. sys.dirname(filename))
		if paths.filep(filename) then
		  os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
		end
		print('Saving network to '..filename)
		torch.save(filename, big_model)		
	end
end

if mode == 'train' then
	init()
	log_file_open:write('-----------------train------------------\n')	
	setup_network(labix, peopleCountForTrain)
	train(trainIterations,labix)
	return 1
end

function scandir(directory)
    local i, t, popen = 0, {}, io.popen
    for filename in popen('ls -a '..directory):lines() do
        i = i + 1
        t[i] = filename
    end
    return t
end


function mysplit(inputstr, sep)
   if sep == nil then
          sep = "%s"
   end
   local t={} ; i=1
   for str in string.gmatch(inputstr, "([^"..sep.."]+)") do
          t[i] = str
          i = i + 1
   end
   return t
end


if mode == 'valid' then
	init()
	log_file_open:write('-----------------validate------------------\n')
	model_lists = scandir(save_train_network_dir..'/lab'.. labix ..'_*.net')
	best_rmse = 1000
	best_rmse_ix = 0
	best_model = nil	
	for modelix, model_lists_item in ipairs(model_lists) do	
		print (modelix .. ' ' .. model_lists_item)	
		load_network(labix, model_lists_item, peopleCountForValidate)
		rmse_i = regress(data, big_model)
		if rmse_i < best_rmse then
			best_rmse = rmse_i
			best_rmse_ix = modelix			
			best_model = big_model:clone()
		end
		log_file_open:flush()
	end
	local best_valid_filename = save_valid_network_dir .. '/' .. paths.basename(model_lists[best_rmse_ix])
	os.execute('mkdir -p ' .. sys.dirname(best_valid_filename))
	if paths.filep(best_valid_filename) then
	  os.execute('mv ' .. best_valid_filename .. ' ' .. best_valid_filename .. '.old')
	end
	print('Saving network to '..best_valid_filename)
	log_file_open:write('Saving network to ')
	log_file_open:write(best_valid_filename)
	log_file_open:write('\n')

	torch.save(best_valid_filename, best_model)
	log_file_open:flush()
end

if mode == 'test' then
	init()
	log_file_open:write('-----------------test------------------\n')
	model_lists = scandir(save_valid_network_dir..'/lab'.. labix ..'*.net')
	for modelix, model_lists_item in ipairs(model_lists) do	
		print (modelix .. ' ' .. model_lists_item)	
		load_network(labix, model_lists_item, peopleCountForValidate)
		rmse_i = regress(data, big_model)
		print(rmse_i)
	end
end

log_file_open:write('-----------------done------------------\n\n')
log_file_open:close()

