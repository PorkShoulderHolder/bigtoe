require 'torch' 
require 'math'
require 'pl'
require 'lfs'

local json = require('cjson')
local prefix = lfs.currentdir() .. "/data/"
local spacing = 15 -- seconds

function loadjsonfile( filename )
	print("loading from " .. prefix .. filename)
	local f = assert(io.open(prefix .. filename))
	local json_str = f:read("*all")
	f:close()
	datatable = json.decode(json_str)
	return datatable
end

function savejsonfile( data, filename )
	print("saving to " .. prefix .. filename)
	local f = assert(io.open(prefix .. filename))
	local json_str = json.encode(data)
	f:write(json_str)
	f:close()
end

function pad( data, padding )
	local out = torch.Tensor(data:size(1) + 2 * padding):fill(0)
	out[{{padding, data:size(1) + padding - 1}}] = data:clone()
	return out
end

function build_tables(bgs, acts, locs)
	
	act_keys = {stationary=1, stationaryautomotive=2, automotive=3, 
				walking=4, running=5, cycling=6}

	local bg_tensor = {}
	local act_tensor = {}
	local loc_tensor = {}
	local testor = {}
	
	for i,t in pairs(locs) do 
		--local o = json.decode(t)
		loc_tensor[i] = {t["unix_date"], t["latitude"], t["longitude"]}
	end

	for i,t in pairs(bgs) do 
		bg_tensor[i] = {t["unix_date"], t["reading"]}
	end

	for i,t in pairs(acts) do 
		act_tensor[i] = {t["unix_date"], act_keys[t["activity"]]}
		testor[t["activity"]] = 0
	end

	local a,i = torch.sort(torch.Tensor(bg_tensor):select(2,1))
	local b,j = torch.sort(torch.Tensor(act_tensor):select(2,1))
	local c,l = torch.sort(torch.Tensor(loc_tensor):select(2,1))	

	bg_tensor = torch.Tensor(bg_tensor)
	loc_tensor = torch.DoubleTensor(loc_tensor)
	act_tensor = torch.Tensor(act_tensor)

	bg_tensor_copy = bg_tensor:clone()
	loc_tensor_copy = loc_tensor:clone()
	act_tensor_copy = act_tensor:clone()

	for k=1,bg_tensor_copy:size(1) do
		bg_tensor_copy[k] = bg_tensor[i[k]]
	end	
	k = 1
	for k=1,loc_tensor_copy:size(1) do
		loc_tensor_copy[k] = loc_tensor[l[k]]
	end	

	for k=1,act_tensor_copy:size(1) do
		act_tensor_copy[k] = act_tensor[j[k]]
	end	
	return bg_tensor_copy, act_tensor_copy, loc_tensor_copy
end	



function form_impulse_tensor(t, start, range, freq)
	--
	-- returns 't' equally spaced by 'freq' seconds
	--
	local t_diff = range
	local freq = freq or 300
	local count = t_diff / freq
	local out_impulse = torch.zeros(count)
	local out = torch.zeros(count, t:size(2) - 1)
	local ind_s = 1
	for i = 1, count do
		if ind_s > t:size(1) then
			goto exit
		end
		if i * freq > t[ind_s][1] - start then
			out[i] = t[ind_s][2]
			out_impulse[i] = 1
			ind_s = ind_s + 1
		end
	end
	::exit::
	return out
end

function form_dense_repr(data)
	--
	-- inverse operation of impulse tensor 
	--
	local data_table = {}
	local ind = 0
	for x, tab in pairs(data) do
		local sum = 0;
		for y, val in pairs(tab) do
			if y > 1 then sum = sum + val end
		end
		if sum != 0 then 
			ind = ind + 1
			data_table[ind] = tab
		end
	end
	return data_table
end

local args = { ... }

function load_data(identifier)

	local locations = loadjsonfile("locations.json")
	print("loaded locations")
	--local transactions = loadjsonfile("/Users/sam.royston/PycharmProjects/PankyV0/data/backup/dump/diabetes/transactions.json")
	--print("loaded transactions")
	local bgs = loadjsonfile("bgs.json")
	print("loaded bgs")
	local activities = loadjsonfile("activities.json")
	print("loaded activities")
	
	return build_tables(bgs, activities, locations)
end

if(path.exists(prefix .. 'bgdata') == false or path.exists(prefix .. 'actdata') == false or path.exists(prefix .. 'locdata' .. args[1]) == false) then
	bgs, acts, locs = load_data(args[1])
	torch.save(prefix .. 'bgdata',bgs)
	torch.save(prefix .. 'locdata',locs)
	torch.save(prefix .. 'actdata',acts)
else
	bgs = torch.load(prefix .. 'bgdata')
	locs = torch.load(prefix .. 'locdata' .. args[1])
	acts = torch.load(prefix .. 'actdata')
end

function get_range(t) return t[t:size(1)][1] - t[1][1] end

function find_timeline(bgs, acts, locs)
	local fins = {bgs[bgs:size(1)][1], acts[acts:size(1)][1], locs[locs:size(1)][1]}
	local starts = {bgs[1][1], acts[1][1], locs[1][1]}
	local finish = math.max(unpack(fins))
	local start = math.min(unpack(starts))
	return start, finish - start
end

local start, range = find_timeline(bgs, acts, locs)


return form_impulse_tensor(bgs, start, range, spacing):cat(form_impulse_tensor(locs, start, range, spacing)):cat(form_impulse_tensor(acts, start, range, spacing))




