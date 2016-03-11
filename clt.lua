require 'torch'
require 'nn'
require('fakecuda').init(true)
require 'lfs'
require 'categorical'
require 'kernelutil'
args = { ... }


function test(identifier)
	identifier = identifier or ''
	local filename = os.date("%c", os.time()):gsub(' ','-') .. (args[2] or '')
	
	local rrr = nn.Sequential()
	local test_net = nn.KernelNet()
	print(test_net)



	local training_data, validation_data = train_test_split(identifier)
	local valid_score = 10000000
	--test_net:save(filename,test_net)
	--test_net:predict(validation_data)

	for i=1,500 do
		test_net:train(training_data, i)
		local vs = test_net:predict(validation_data)
		if vs < valid_score then
			valid_score = vs 
			test_net:save(filename)
		end
		test_net:updateVisuals()
	end
end

function apply_to_all(dir_name, fn)
	--
	--	apply function to all networks saved in directory
	--
	for file in lfs.dir(dir_name) do 
		if file:find(".t7") ~= nil then
			fn(dir_name .. '/' .. file)
		end
	end
end

function valid_network(net_file)
	local model = torch.load(net_file)
	local training_data, validation_data = train_test_split('')
	print("loaded: " .. net_file)
	print(model)
	model:predict(validation_data)
end

if args[1] == 'train' then
	test(args[2])
elseif args[1] == "get_weights" then
	local net_file = args[2]
	local model = torch.load(net_file)
	local fn = net_file:gsub(".net","")
	model:saveWeights(fn)
elseif args[1] == "valid" then
	if args[2] ~= "all"	then
		valid_network(args[2])
	else
		apply_to_all("experiments", valid_network)
	end
elseif args[1] == "viz" then
	local model = torch.load(args[2])
	model:updateVisuals()
end


