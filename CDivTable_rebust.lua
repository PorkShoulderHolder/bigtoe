local CDivTable_robust, parent = torch.class('CDivTable_robust', 'nn.Module')

function CDivTable_robust:__init()
   parent.__init(self)
   self.gradInput = {}
end

function CDivTable_robust:updateOutput(input)
   local lower_lim = 20
   
   self.output:resizeAs(input[1]):copy(input[1])   
   self.output:cdiv(input[2])
   
   self.output[input[2]:abs():le(10)] = 0.0
   -- self.output[self.output:ne(self.output)] = 0.0
   return self.output
end


function CDivTable_robust:updateGradInput(input, gradOutput)


   self.gradInput[1] = self.gradInput[1] or input[1].new()
   self.gradInput[2] = self.gradInput[2] or input[2].new()

   tmp = input[2]:abs():le(10)

   self.gradInput[1]:resizeAs(input[1]):copy(gradOutput):cdiv(input[2])
   self.gradInput[2]:resizeAs(input[2]):zero():addcdiv(-1,self.gradInput[1],input[2]):cmul(input[1])
   
   self.gradInput[1][tmp:eq(1)] = 0.0 --do not backpropagate nan when you see zero/zero
   self.gradInput[2][tmp:eq(1)] = 0.0
  

   for i=#input+1, #self.gradInput do
       self.gradInput[i] = nil
   end
  
   -- self.gradInput[1][self.gradInput[1]:ne(self.gradInput[1])] = 0.0
   -- self.gradInput[2][self.gradInput[2]:ne(self.gradInput[2])] = 0.0


   return self.gradInput
end