require 'nn'
local withCuda = pcall(require, 'cutorch')

require 'libstn'
if withCuda then
   require 'libcustn'
end

require('stn.AffineTransformMatrixGenerator')
require('stn.AffineGridGeneratorBHWD')
require('stn.BilinearSamplerBHWD')

require('stn.L1DistanceBatchMat')
require('stn.BatchDiscrimination')

require('stn.test')

return nn
