require 'stn'

if true then
   bs = 5
   nInputPlane = 4
   nkernels = 8
   kerneldim = 3
   m = nn.BatchDiscrimination(nInputPlane,nkernels,kerneldim)
   input = torch.Tensor(bs,nInputPlane):uniform()
   for i=2, bs do
      input[i]:copy(input[1])
   end
   m:forward(input)
   print(m.output)
   m:backward(input, m.output:clone():uniform())
   
   jac = nn.Jacobian
   err=jac.testJacobian(m, input)
   print(err)
   
end

if true then
   
   bs = 24
   d = 17
   e = 27
   input=torch.Tensor(bs,d, e):uniform()
   inputC = input:cuda()
   
   m = nn.L1DistanceBatchMat()
   m:updateOutput(input)
   --print(m.output)
   grad = m.output:clone():uniform()
   gradC = grad:cuda()
   
   mC = nn.L1DistanceBatchMat():cuda()
   
   outputF = m:updateOutput(input)
   outputC = mC:updateOutput(inputC)
   
   print('fwd relative error : ', (outputC:double()-outputF:double()):abs():sum()/outputC:max())
   
   gradInputF = m:updateGradInput(input, grad)
   gradInputC = mC:updateGradInputGpu(inputC, gradC)
   
   print('bwd relative error : ', (gradInputC:double()-gradInputF:double()):sum()/gradInputC:max())
   print('bwd relative error : ', (gradInputC:double()-gradInputF:double()):abs():sum()/gradInputC:max())
   
end


if true then
   bs = 64
   nInputPlane = 1024
   nkernels = 50
   kerneldim = 5
   m = nn.BatchDiscrimination(nInputPlane,nkernels,kerneldim):cuda()
   input = torch.Tensor(bs,nInputPlane):uniform():cuda()
   
   m:forward(input)
   g=m.output:clone():uniform()

   n=100
   
   a = torch.tic()
   for i=1,n do   
      m:forward(input)
      cutorch.synchronize()
   end
   print('fwd time :', torch.toc(a)/n)

   a = torch.tic()
   for i=1,n do   
      m:backward(input, g)
      cutorch.synchronize()
   end
   print('bwd time :', torch.toc(a)/n)

end

