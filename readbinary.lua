require 'torch'

function fromfile(fname)
   local file = io.open(fname .. '.type')
   local type = file:read('*all')

   local x
   if type == 'float32' then
      x = torch.FloatTensor(torch.FloatStorage(fname))
   elseif type == 'int32' then
      x = torch.IntTensor(torch.IntStorage(fname))
   elseif type == 'int64' then
      x = torch.LongTensor(torch.LongStorage(fname))
   else
      print(fname, type)
      assert(false)
   end
   local file = io.open(fname .. '.dim')
   local dim = {}
   for line in file:lines() do
      table.insert(dim, tonumber(line))
   end
   x = x:reshape(torch.LongStorage(dim))
   return x:float()
end

x = fromfile('../diagnosisFromLabs/data/valid/valid_lab_to_icd9_normalized.bin')
return x
