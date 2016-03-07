require 'torch'
require 'nn'
require 'cunn'
require 'lfs'
require 'categorical'

args = { ... }


function train_test_split(identifier)
	-- body
	x,y,z = assert(loadfile('readjson.lua'))(identifier) -- include cluster partition identifier to specify which

	local t_split = 5.0 / 6.0
	local tv_split = 9.0 / 10.0

	x = x[{{1, x:size(1) * t_split}}]:clone()
--	y = y[{{1, y:size(1) * t_split}}]:clone()
--	z = z[{{1, z:size(1) * t_split}}]:clone()
	local training_data = x[{{1,x:size(1) * tv_split}}]:clone() --, y[{{1, y:size(1) * tv_split}}]:clone(), z[{{1, z:size(1) * tv_split}}]:clone()}
	local validation_data = x[{{x:size(1) * tv_split, x:size(1)}}]:clone() -- y[{{y:size(1) * tv_split, y:size(1)}}]:clone()	, z[{{z:size(1) * tv_split, z:size(1)}}]:clone()}

	return training_data, validation_data
end

function test(identifier)
	identifier = identifier or ''
	local filename = os.date("%c", os.time()):gsub(' ','-') .. (args[2] or '')
	
	local rrr = nn.Sequential()
	local test_net = nn.KernelNet()
	print(test_net)

	test_net = test_net:cuda()


	local training_data, validation_data = train_test_split(identifier)
	local valid_score = 10000000
	test_net:save(filename,test_net)
	print("saved!")
	test_net:predict(validation_data)

	for i=1,150 do
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
	test()
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
end

