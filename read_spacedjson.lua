require 'torch' 
require 'math'
require 'pl'
require 'lfs'
require 'kernelutil'

local json = require('cjson')
local prefix = lfs.currentdir() .. "/data/"


function loadjson_asarray( filename )
	print(filename)
	local f = assert(io.open(filename))
	local json_str = f:read("*all")
	f:close()
	local datatable = json.decode(json_str)

	local out = torch.Tensor(datatable)
	return out
end


function test( )
	local data = loadjson_asarray( arg[1] )
	print(data:size())
	local training, testing = tts(data) 
end

