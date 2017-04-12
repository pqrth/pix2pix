require 'nngraph'
require 'image'
require 'Binarize'

function defineG_encoder_decoder(input_nc, output_nc, ngf)
    local netG = nil 
    -- input is (nc) x 256 x 256
    local e1 = - nn.SpatialConvolution(input_nc, ngf, 4, 4, 2, 2, 1, 1)
    -- input is (ngf) x 128 x 128
    local e2 = e1 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)
    -- input is (ngf * 2) x 64 x 64
    local e3 = e2 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 2, ngf * 4, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 4)
    -- input is (ngf * 4) x 32 x 32
    local e4 = e3 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 4, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 16 x 16
    local e5 = e4 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 8 x 8
    local e6 = e5 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 4 x 4
    local e7 = e6 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 2 x 2
    local e8 = e7 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 1 x 1
    
    local d1 = e8 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
    -- input is (ngf * 8) x 2 x 2
    local d2 = d1 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
    -- input is (ngf * 8) x 4 x 4
    local d3 = d2 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
    -- input is (ngf * 8) x 8 x 8
    local d4 = d3 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 16 x 16
    local d5 = d4 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8, ngf * 4, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 4)
    -- input is (ngf * 4) x 32 x 32
    local d6 = d5 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 4, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)
    -- input is (ngf * 2) x 64 x 64
    local d7 = d6 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2, ngf, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf)
    -- input is (ngf) x128 x 128
    local d8 = d7 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf, output_nc, 4, 4, 2, 2, 1, 1)
    -- input is (nc) x 256 x 256
    
    local o1 = d8 - nn.Tanh()
    
    netG = nn.gModule({e1},{o1})

    return netG
end

function defineG_unet(input_nc, output_nc, ngf)
    local netG = nil
    -- input is (nc) x 256 x 256
    local e1 = - nn.SpatialConvolution(input_nc, ngf, 4, 4, 2, 2, 1, 1)
    -- input is (ngf) x 128 x 128
    local e2 = e1 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)
    -- input is (ngf * 2) x 64 x 64
    local e3 = e2 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 2, ngf * 4, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 4)
    -- input is (ngf * 4) x 32 x 32
    local e4 = e3 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 4, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 16 x 16
    local e5 = e4 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 8 x 8
    local e6 = e5 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 4 x 4
    local e7 = e6 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 2 x 2
    local e8 = e7 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 1 x 1
    
    local d1_ = e8 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
    -- input is (ngf * 8) x 2 x 2
    local d1 = {d1_,e7} - nn.JoinTable(2)
    local d2_ = d1 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
    -- input is (ngf * 8) x 4 x 4
    local d2 = {d2_,e6} - nn.JoinTable(2)
    local d3_ = d2 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
    -- input is (ngf * 8) x 8 x 8
    local d3 = {d3_,e5} - nn.JoinTable(2)
    local d4_ = d3 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 16 x 16
    local d4 = {d4_,e4} - nn.JoinTable(2)
    local d5_ = d4 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 4, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 4)
    -- input is (ngf * 4) x 32 x 32
    local d5 = {d5_,e3} - nn.JoinTable(2)
    local d6_ = d5 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 4 * 2, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)
    -- input is (ngf * 2) x 64 x 64
    local d6 = {d6_,e2} - nn.JoinTable(2)
    local d7_ = d6 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2 * 2, ngf, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf)
    -- input is (ngf) x128 x 128
    local d7 = {d7_,e1} - nn.JoinTable(2)
    local d8 = d7 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2, output_nc, 4, 4, 2, 2, 1, 1)
    -- input is (nc) x 256 x 256
    
    local o1 = d8 - nn.Tanh()
    
    netG = nn.gModule({e1},{o1})
    
    --graph.dot(netG.fg,'netG')
    
    return netG
end

function defineG_unet_exposure(input_nc, output_nc, ngf)
    local netG = nil
    local input = - nn.Identity()
    -- input is (nc) x 256 x 256
    local e1 = input - nn.SpatialConvolution(input_nc, ngf, 4, 4, 2, 2, 1, 1)
    -- input is (ngf) x 128 x 128
    local e2 = e1 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)
    -- input is (ngf * 2) x 64 x 64
    local e3 = e2 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 2, ngf * 4, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 4)
    -- input is (ngf * 4) x 32 x 32
    local e4 = e3 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 4, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 16 x 16
    local e5 = e4 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 8 x 8
    local e6 = e5 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 4 x 4
    local e7 = e6 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 2 x 2
    local e8 = e7 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 1 x 1

    local d1_ = e8 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
    -- input is (ngf * 8) x 2 x 2
    local d1 = {d1_,e7} - nn.JoinTable(2)
    local d2_ = d1 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
    -- input is (ngf * 8) x 4 x 4
    local d2 = {d2_,e6} - nn.JoinTable(2)
    local d3_ = d2 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
    -- input is (ngf * 8) x 8 x 8
    local d3 = {d3_,e5} - nn.JoinTable(2)
    local d4_ = d3 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 16 x 16
    local d4 = {d4_,e4} - nn.JoinTable(2)
    local d5_ = d4 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 4, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 4)
    -- input is (ngf * 4) x 32 x 32
    local d5 = {d5_,e3} - nn.JoinTable(2)
    local d6_ = d5 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 4 * 2, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)
    -- input is (ngf * 2) x 64 x 64
    local d6 = {d6_,e2} - nn.JoinTable(2)
    local d7_ = d6 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2 * 2, ngf, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf)
    -- input is (ngf) x128 x 128
    local d7 = {d7_,e1} - nn.JoinTable(2)
    local d8_ = d7 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2, output_nc, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(output_nc)
    -- input is (nc) x 256 x 256

    local input_deprocess = input - nn.AddConstant(1) - nn.MulConstant(0.5)  -- deprocess input image [-1,1] to [0,1]
    local d8 = {d8_,input_deprocess} - nn.JoinTable(2)
    local d9_ = d8 - nn.ReLU(true) - nn.SpatialConvolution(output_nc * 2, output_nc, 3, 3, 1, 1, 1, 1) - nn.ReLU(true)
    local shadowMap = {d9_,input_deprocess} - nn.CAddTable() - nn.AddConstant(0.00001) - nn.HardTanh()   -- already >=input_deprocess>=0, so hardTanh produces between [input_deprocess,1]
    local shadowMapInv = shadowMap - nn.Power(-1)  -- [1, 1/input_deprocess]

    local output = {input_deprocess,shadowMapInv} - nn.CMulTable()

    local output_ = output - nn.MulConstant(2) - nn.AddConstant(-1)  -- clamp between [0,1] and process output cleaned image [0,1] to [-1,1]
    local shadowMap_ = shadowMap - nn.MulConstant(2) - nn.AddConstant(-1)

    netG = nn.gModule({input},{output_,shadowMap_})

    --graph.dot(netG.fg,'netG')

    return netG
end


function defineG_unet_exposure_simple(input_nc, output_nc, ngf)
    local netG = nil
    local input = - nn.Identity()
    -- input is (nc) x 256 x 256
    local e1 = input - nn.SpatialConvolution(input_nc, ngf, 4, 4, 2, 2, 1, 1)
    -- input is (ngf) x 128 x 128
    local e2 = e1 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)
    -- input is (ngf * 2) x 64 x 64
    local e3 = e2 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 2, ngf * 4, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 4)
    -- input is (ngf * 4) x 32 x 32
    local e4 = e3 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 4, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 16 x 16
    local e5 = e4 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 8 x 8
    local e6 = e5 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 4 x 4
    local e7 = e6 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 2 x 2
    local e8 = e7 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 1 x 1

    local d1_ = e8 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
    -- input is (ngf * 8) x 2 x 2
    local d1 = {d1_,e7} - nn.JoinTable(2)
    local d2_ = d1 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
    -- input is (ngf * 8) x 4 x 4
    local d2 = {d2_,e6} - nn.JoinTable(2)
    local d3_ = d2 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
    -- input is (ngf * 8) x 8 x 8
    local d3 = {d3_,e5} - nn.JoinTable(2)
    local d4_ = d3 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 16 x 16
    local d4 = {d4_,e4} - nn.JoinTable(2)
    local d5_ = d4 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 4, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 4)
    -- input is (ngf * 4) x 32 x 32
    local d5 = {d5_,e3} - nn.JoinTable(2)
    local d6_ = d5 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 4 * 2, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)
    -- input is (ngf * 2) x 64 x 64
    local d6 = {d6_,e2} - nn.JoinTable(2)
    local d7_ = d6 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2 * 2, ngf, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf)
    -- input is (ngf) x128 x 128
    local d7 = {d7_,e1} - nn.JoinTable(2)
    local d8 = d7 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2, output_nc, 4, 4, 2, 2, 1, 1) - nn.ReLU(true)
    -- input is (nc) x 256 x 256

    --local o1 = d8 - nn.Tanh()
    local d8_ = d8 - nn.AddConstant(1)
    local input_deprocess = input - nn.AddConstant(1) - nn.MulConstant(0.5)
    local o1_ = {input_deprocess,d8_} - nn.CMulTable()
    local o2 = o1_ - nn.MulConstant(2) - nn.AddConstant(-1)
    local clippedOutput = o1_ - nn.HardTanh() - nn.MulConstant(2) - nn.AddConstant(-1)
    local shadowMap = d8_ - nn.Power(-1) - nn.MulConstant(2) - nn.AddConstant(-1)
    netG = nn.gModule({input},{o2,clippedOutput,shadowMap})

    --graph.dot(netG.fg,'netG')

    return netG
end


--[[
A custom convolution layer for performing LOG.
This is a fixed layer i.e. weights do not update.
]]--
do
    -- override init to set appropriate constraints on dimensions
    local LaplacianConv, parent = torch.class('nn.LaplacianConv', 'nn.SpatialConvolution')
      function LaplacianConv:__init(nInputPlane, nOutputPlane, k)      
        k = k or 3
        if k < 3 then k = 3 end
        pad = math.ceil((k-1)/2)
        parent.__init(self,nInputPlane, nOutputPlane, k, k, 1, 1, pad, pad)
      end

      -- overide reset() to set weights as LOG kernel
      function LaplacianConv:reset()
        if self.bias then
          self.bias:fill(0)
        end
        if self.weight then
          self.weight:fill(0)
          local lp = image.laplacian(self.weight:size()[3],nil,nil,true)
          for i=1,self.weight:size()[1] do
            self.weight[i][i]:copy(lp)
          end 
        end
      end
    
      -- empty accGradParameters() to prevent any weight updates
      function LaplacianConv:accGradParameters(input, gradOutput, scale)
      end
end

--[[
A custom convolution layer for Gaussian filter.
This is a fixed layer i.e. weights do not update.
]]--
do
    -- override init to set appropriate constraints on dimensions
    local GaussianConv, parent = torch.class('nn.GaussianConv', 'nn.SpatialConvolution')
      function GaussianConv:__init(nInputPlane, nOutputPlane, k)
        k = k or 3
        if k < 3 then k = 3 end
        pad = math.ceil((k-1)/2)
        parent.__init(self,nInputPlane, nOutputPlane, k, k, 1, 1, pad, pad)
      end

      -- overide reset() to set weights as Gaussian kernel
      function GaussianConv:reset()
        if self.bias then
          self.bias:fill(0)
        end
        if self.weight then
          self.weight:fill(0)
          local lp = image.gaussian(self.weight:size()[3],nil,nil,true)
          for i=1,self.weight:size()[1] do
            self.weight[i][i]:copy(lp)
          end
        end
      end

      -- empty accGradParameters() to prevent any weight updates
      function GaussianConv:accGradParameters(input, gradOutput, scale)
      end
end

--[[
A custom convolution layer for applying Sobel filter in X-direction.
This is a fixed layer i.e. weights do not update.
]]--
do
    local SobelXConv, parent = torch.class('nn.SobelXConv', 'nn.SpatialConvolution')
      -- override init to set appropriate constraints on dimensions
      function SobelXConv:__init(nInputPlane, nOutputPlane)
        parent.__init(self,nInputPlane, nOutputPlane, 3, 3, 1, 1, 1, 1)
      end

      -- overide reset() to set weights as Sobel X kernel
      function SobelXConv:reset()
        if self.bias then
          self.bias:fill(0)
        end
        if self.weight then
          self.weight:fill(0)
          local sobel_x = torch.DoubleTensor(3,3):fill(0)
          sobel_x[1][1] = -1.0/8
          sobel_x[2][1] = -2.0/8
          sobel_x[3][1] = -1.0/8
          sobel_x[1][3] = 1.0/8
          sobel_x[2][3] = 2.0/8
          sobel_x[3][3] = 1.0/8
          
          for i=1,self.weight:size()[1] do
            self.weight[i][i]:copy(sobel_x)
          end
        end
      end

      -- empty accGradParameters() to prevent any weight updates
      function SobelXConv:accGradParameters(input, gradOutput, scale)
      end
end


--[[
A custom convolution layer for applying Sobel filter in Y-direction.
This is a fixed layer i.e. weights do not update.
]]-- 
do
    local SobelYConv, parent = torch.class('nn.SobelYConv', 'nn.SpatialConvolution')
      -- override init to set appropriate constraints on dimensions
      function SobelYConv:__init(nInputPlane, nOutputPlane)
        parent.__init(self,nInputPlane, nOutputPlane, 3, 3, 1, 1, 1, 1)
      end

      -- overide reset() to set weights as Sobel Y kernel
      function SobelYConv:reset()
        if self.bias then
          self.bias:fill(0)
        end
        if self.weight then
          self.weight:fill(0)
          local sobel_y = torch.DoubleTensor(3,3):fill(0)
          sobel_y[1][1] = 1.0/8
          sobel_y[1][2] = 2.0/8
          sobel_y[1][3] = 1.0/8
          sobel_y[3][1] = -1.0/8
          sobel_y[3][2] = -2.0/8
          sobel_y[3][3] = -1.0/8
          
          for i=1,self.weight:size()[1] do
            self.weight[i][i]:copy(sobel_y)
          end
        end
      end

      -- empty accGradParameters() to prevent any weight updates
      function SobelYConv:accGradParameters(input, gradOutput, scale)
      end
end

--[[
A custom convolution laye.
This is a fixed layer i.e. weights do not update.
]]--
do
    -- override init to set appropriate constraints on dimensions
    local FixedLayerWise2DConv, parent = torch.class('nn.FixedLayerWise2DConv', 'nn.SpatialConvolution')
      function FixedLayerWise2DConv:__init(nInputPlane, nOutputPlane, kernel2D)
        k = kernel2D:size()[1]
	pad = math.ceil((k-1)/2)
	self.kernel2D = kernel2D
        parent.__init(self,nInputPlane, nOutputPlane, k, k, 1, 1, pad, pad)
      end

      -- overide reset() to set weights as custom kernel
      function FixedLayerWise2DConv:reset()
        if self.bias then
          self.bias:fill(0)
        end
        if self.weight then
          self.weight:fill(0)
          for i=1,self.weight:size()[1] do
            self.weight[i][i]:copy(self.kernel2D)
          end
        end
      end

      -- empty accGradParameters() to prevent any weight updates
      function FixedLayerWise2DConv:accGradParameters(input, gradOutput, scale)
      end
end


do
    local IdentityBlockGrad, parent = torch.class('nn.IdentityBlockGrad', 'nn.Identity')
      function IdentityBlockGrad:__init()
	parent.__init(self)
      end

      function IdentityBlockGrad:updateGradInput(input, gradOutput)
	self.gradInput = gradOutput:clone():fill(0)
	return self.gradInput
      end
end


--[[
exposure model based generator
returns-
o2: processed(cleaned) image
shadowMap_: predicted shadow
shadowLaplacian_: LOG of predicted shadow map
shadowSobel_: map of magnitude of sobel filter filter output on shadow map 
]]--
function defineG_unet_exposure_shadow_map2(input_nc, output_nc, ngf)
    local netG = nil
    local input = - nn.Identity()
    -- input is (nc) x 256 x 256
    local e1 = input - nn.SpatialConvolution(input_nc, ngf, 4, 4, 2, 2, 1, 1)
    -- input is (ngf) x 128 x 128
    local e2 = e1 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)
    -- input is (ngf * 2) x 64 x 64
    local e3 = e2 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 2, ngf * 4, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 4)
    -- input is (ngf * 4) x 32 x 32
    local e4 = e3 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 4, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 16 x 16
    local e5 = e4 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 8 x 8
    local e6 = e5 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 4 x 4
    local e7 = e6 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 2 x 2
    local e8 = e7 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 1 x 1

    local d1_ = e8 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
    -- input is (ngf * 8) x 2 x 2
    local d1 = {d1_,e7} - nn.JoinTable(2)
    local d2_ = d1 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
    -- input is (ngf * 8) x 4 x 4
    local d2 = {d2_,e6} - nn.JoinTable(2)
    local d3_ = d2 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
    -- input is (ngf * 8) x 8 x 8
    local d3 = {d3_,e5} - nn.JoinTable(2)
    local d4_ = d3 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 16 x 16
    local d4 = {d4_,e4} - nn.JoinTable(2)
    local d5_ = d4 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 4, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 4)
    -- input is (ngf * 4) x 32 x 32
    local d5 = {d5_,e3} - nn.JoinTable(2)
    local d6_ = d5 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 4 * 2, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)
    -- input is (ngf * 2) x 64 x 64
    local d6 = {d6_,e2} - nn.JoinTable(2)
    local d7_ = d6 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2 * 2, ngf, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf)
    -- input is (ngf) x128 x 128
    local d7 = {d7_,e1} - nn.JoinTable(2)
    local d8_ = d7 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2, output_nc, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(output_nc)
    -- input is (nc) x 256 x 256

    local input_deprocess = input - nn.AddConstant(1) - nn.MulConstant(0.5)  -- deprocess input image [-1,1] to [0,1]
    local d8 = {d8_,input_deprocess} - nn.JoinTable(2)
    local d9_ = d8 - nn.ReLU(true) - nn.SpatialConvolution(output_nc * 2, output_nc, 3, 3, 1, 1, 1, 1) - nn.ReLU(true)
    local shadowMap = {d9_,input_deprocess} - nn.CAddTable() - nn.AddConstant(0.00001) - nn.HardTanh()   -- already >=input_deprocess>=0, so hardTanh produces between [input_deprocess,1]
    local d9 = shadowMap - nn.Power(-1)  -- [1, 1/input_deprocess]

    local o1_ = {input_deprocess,d9} - nn.CMulTable()
    local o2 = o1_ - nn.MulConstant(2) - nn.AddConstant(-1)  -- clamp between [0,1] and process output cleaned image [0,1] to [-1,1]

    local correlation_window_size = 5
    local ave_kernel = torch.Tensor(correlation_window_size,correlation_window_size):fill(1/(correlation_window_size*correlation_window_size))
    local mean_windowed_shadow = shadowMap - nn.FixedLayerWise2DConv(output_nc,output_nc,ave_kernel)
    local mean_windowed_output = o1_ - nn.FixedLayerWise2DConv(output_nc,output_nc,ave_kernel)
    local mean1_X_mean2 = {mean_windowed_shadow, mean_windowed_output} - nn.CMulTable()
    local windowed_cross_corr_shadow_output = {shadowMap,o1_} - nn.CMulTable() - nn.FixedLayerWise2DConv(output_nc,output_nc,ave_kernel)
    local windowed_CCorr_centered_shadow_output = {windowed_cross_corr_shadow_output, mean1_X_mean2} - nn.CSubTable()

--[[local shadow_sq_windowed_mean = shadowMap - nn.Square() - nn.FixedLayerWise2DConv(output_nc,output_nc,ave_kernel)
    local output_sq_windowed_mean = o1_ - nn.Square() - nn.FixedLayerWise2DConv(output_nc,output_nc,ave_kernel)
    local mean_windowed_shadow_sq = mean_windowed_shadow - nn.Square()
    local mean_windowed_output_sq = mean_windowed_output - nn.Square()
    local shadow_std = {shadow_sq_windowed_mean,mean_windowed_shadow_sq} - nn.CSubTable() - nn.ReLU(true) - nn.Sqrt()
    local output_std = {output_sq_windowed_mean,mean_windowed_output_sq} - nn.CSubTable() - nn.ReLU(true) - nn.Sqrt()
    local S1_X_S2 = {shadow_std,output_std} - nn.CMulTable() - nn.AddConstant(0.000001)
    local windowed_NGC_shadow_output = {windowed_CCorr_centered_shadow_output,S1_X_S2} - nn.CDivTable()
]]--

    local downsampledShadow = shadowMap - nn.SpatialAveragePooling(4,4,4,4)
    local downsampledOutput = o1_ - nn.SpatialAveragePooling(4,4,4,4)
    local dw_mean_windowed_shadow = downsampledShadow - nn.FixedLayerWise2DConv(output_nc,output_nc,ave_kernel)
    local dw_mean_windowed_output = downsampledOutput - nn.FixedLayerWise2DConv(output_nc,output_nc,ave_kernel)
    local dw_mean1_X_mean2 = {dw_mean_windowed_shadow, dw_mean_windowed_output} - nn.CMulTable()
    local dw_windowed_cross_corr_shadow_output = {downsampledShadow,downsampledOutput} - nn.CMulTable() - nn.FixedLayerWise2DConv(output_nc,output_nc,ave_kernel)
    local dw_windowed_CCorr_centered_shadow_output = {dw_windowed_cross_corr_shadow_output, dw_mean1_X_mean2} - nn.CSubTable()

    local shadowSobelX_2 = shadowMap - nn.SobelXConv(output_nc,output_nc) - nn.Abs() -- nn.Square()
    local shadowSobelY_2 = shadowMap - nn.SobelYConv(output_nc,output_nc) - nn.Abs() -- nn.Square()
    local shadowSobel = {shadowSobelX_2,shadowSobelY_2} - nn.CAddTable() - nn.MulConstant(0.5) -- nn.Sqrt() - nn.MulConstant(0.71) -- nn.Power(0.3)

    local dw_shadowSobelX_2 = downsampledShadow - nn.SobelXConv(output_nc,output_nc) - nn.Abs() -- nn.Square()
    local dw_shadowSobelY_2 = downsampledShadow - nn.SobelYConv(output_nc,output_nc) - nn.Abs() -- nn.Square()
    local dw_shadowSobel = {dw_shadowSobelX_2,dw_shadowSobelY_2} - nn.CAddTable() - nn.MulConstant(0.5) -- nn.Sqrt() - nn.MulConstant(0.71) -- nn.Power(0.3)

    local CCorr_pow = windowed_CCorr_centered_shadow_output - nn.Abs() -- nn.Sqrt() - nn.Sqrt() - nn.IdentityBlockGrad()
    local dw_CCorr_pow = dw_windowed_CCorr_centered_shadow_output - nn.Abs() -- nn.Sqrt() - nn.Sqrt() - nn.IdentityBlockGrad()
    local sobelScaled = {shadowSobel,CCorr_pow} - nn.CMulTable() - nn.Sqrt()
    local dw_sobelScaled = {dw_shadowSobel,dw_CCorr_pow} - nn.CMulTable() - nn.Sqrt()
    -- [0,1] to [-1,1]
    local shadowMap_ = shadowMap - nn.MulConstant(2) - nn.AddConstant(-1)
    local windowed_CCorr_centered_shadow_output_ = windowed_CCorr_centered_shadow_output -- nn.Abs() - nn.Sqrt() - nn.Sqrt() - nn.Threshold(0.3,0) - nn.Sqrt()-- nn.MulConstant(2) - nn.AddConstant(-1)
    local shadowSobel_ = sobelScaled - nn.MulConstant(2) - nn.AddConstant(-1)
    local dw_shadowSobel_ = dw_sobelScaled - nn.MulConstant(2) - nn.AddConstant(-1)

    netG = nn.gModule({input},{o2, shadowMap_, windowed_CCorr_centered_shadow_output_, dw_windowed_CCorr_centered_shadow_output, shadowSobel_, dw_shadowSobel_})

    --graph.dot(netG.fg,'netG')

    return netG
end


function defineG_unet_exposure_shadow_masked(input_nc, output_nc, ngf)
    local netG = nil
    local input = - nn.Identity()
    -- input is (nc) x 256 x 256
    local e1 = input - nn.SpatialConvolution(input_nc, ngf, 4, 4, 2, 2, 1, 1)
    -- input is (ngf) x 128 x 128
    local e2 = e1 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)
    -- input is (ngf * 2) x 64 x 64
    local e3 = e2 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 2, ngf * 4, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 4)
    -- input is (ngf * 4) x 32 x 32
    local e4 = e3 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 4, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 16 x 16
    local e5 = e4 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 8 x 8
    local e6 = e5 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 4 x 4
    local e7 = e6 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 2 x 2
    local e8 = e7 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 1 x 1

    local d1_ = e8 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
    -- input is (ngf * 8) x 2 x 2
    local d1 = {d1_,e7} - nn.JoinTable(2)
    local d2_ = d1 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
    -- input is (ngf * 8) x 4 x 4
    local d2 = {d2_,e6} - nn.JoinTable(2)
    local d3_ = d2 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
    -- input is (ngf * 8) x 8 x 8
    local d3 = {d3_,e5} - nn.JoinTable(2)
    local d4_ = d3 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 16 x 16
    local d4 = {d4_,e4} - nn.JoinTable(2)
    local d5_ = d4 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 4, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 4)
    -- input is (ngf * 4) x 32 x 32
    local d5 = {d5_,e3} - nn.JoinTable(2)
    local d6_ = d5 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 4 * 2, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)
    -- input is (ngf * 2) x 64 x 64
    local d6 = {d6_,e2} - nn.JoinTable(2)
    local d7_ = d6 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2 * 2, ngf, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf)
    -- input is (ngf) x128 x 128
    local d7 = {d7_,e1} - nn.JoinTable(2)
    local d8_ = d7 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2, output_nc, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(output_nc)
    -- input is (nc) x 256 x 256

    local input_deprocess = input - nn.AddConstant(1) - nn.MulConstant(0.5)  -- deprocess input image [-1,1] to [0,1]
    local d8 = {d8_,input_deprocess} - nn.JoinTable(2)
    local d9_ = d8 - nn.ReLU(true) - nn.SpatialConvolution(output_nc * 2, 1, 3, 3, 1, 1, 1, 1) - nn.HardTanh() -- nn.ReLU(true)

    local d9_1_ = d9_ - nn.Copy()
    local d9_2_ = d9_ - nn.Copy()
    local d9_3_ = d9_ - nn.Copy()
    local mask3D_soft = {d9_1_,d9_2_,d9_3_} - nn.JoinTable(2)

    local mask2D = d9_ - nn.Binarize(true) - nn.AddConstant(1) - nn.MulConstant(0.5)
    local mask2D_1 = mask2D - nn.Copy()
    local mask2D_2 = mask2D - nn.Copy()
    local mask2D_3 = mask2D - nn.Copy()
    local mask3D = {mask2D_1,mask2D_2,mask2D_3} - nn.JoinTable(2)
    local mask3D_comp = mask3D - nn.AddConstant(-1) - nn.MulConstant(-1)

    local inputDepMasked__ = {input_deprocess,mask3D} - nn.CMulTable()
    local inputDepMasked = {inputDepMasked__,mask3D_comp} - nn.CAddTable() - nn.AddConstant(0.00001) - nn.HardTanh()
    local inputDepMaskedRecip = inputDepMasked - nn.Power(-1)
    local outputmasked = {input_deprocess,inputDepMaskedRecip} - nn.CMulTable()

    local ee1 = input - nn.SpatialConvolution(input_nc, ngf, 4, 4, 2, 2, 1, 1)
    -- input is (ngf) x 128 x 128
    local ee2 = ee1 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)
    -- input is (ngf * 2) x 64 x 64
    local ee3 = ee2 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 2, ngf * 4, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 4)
    -- input is (ngf * 4) x 32 x 32
    local ee4 = ee3 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 4, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 16 x 16
    local ee5 = ee4 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 8 x 8
    local ee6 = ee5 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 4 x 4
    local ee7 = ee6 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 2 x 2
    local ee8 = ee7 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 1 x 1

    local dd1_ = ee8 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
    -- input is (ngf * 8) x 2 x 2
    local dd1 = {dd1_,ee7} - nn.JoinTable(2)
    local dd2_ = dd1 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
    -- input is (ngf * 8) x 4 x 4
    local dd2 = {dd2_,ee6} - nn.JoinTable(2)
    local dd3_ = dd2 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
    -- input is (ngf * 8) x 8 x 8
    local dd3 = {dd3_,ee5} - nn.JoinTable(2)
    local dd4_ = dd3 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 16 x 16
    local dd4 = {dd4_,ee4} - nn.JoinTable(2)
    local dd5_ = dd4 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 4, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 4)
    -- input is (ngf * 4) x 32 x 32
    local dd5 = {dd5_,ee3} - nn.JoinTable(2)
    local dd6_ = dd5 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 4 * 2, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)
    -- input is (ngf * 2) x 64 x 64
    local dd6 = {dd6_,ee2} - nn.JoinTable(2)
    local dd7_ = dd6 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2 * 2, ngf, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf)
    -- input is (ngf) x128 x 128
    local dd7 = {dd7_,ee1} - nn.JoinTable(2)
    local dd8_ = dd7 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2, output_nc, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(output_nc)
    
    local dd8 = {dd8_,input_deprocess} - nn.JoinTable(2)
    local dd9_ = dd8 - nn.ReLU(true) - nn.SpatialConvolution(output_nc * 2, output_nc, 3, 3, 1, 1, 1, 1) - nn.ReLU(true)
    
    local shadowMap = {dd9_,input_deprocess} - nn.CAddTable() - nn.AddConstant(0.00001) - nn.HardTanh()

    --local shadowMap = {d9_,input_deprocess} - nn.CAddTable() - nn.AddConstant(0.00001) - nn.HardTanh()   -- already >=input_deprocess>=0, so hardTanh produces between [input_deprocess,1]
    local shadowMapReciprocal = shadowMap - nn.Power(-1)  -- [1, 1/input_deprocess]

    local output = {input_deprocess,shadowMapReciprocal} - nn.CMulTable()

    local shadowMapBlurred = shadowMap - nn.GaussianConv(output_nc,output_nc,5) - nn.SpatialMaxPooling(5, 5, 1, 1, 2, 2)
    local shadowMapBlurredMasked = {shadowMapBlurred,mask3D_comp} - nn.CMulTable()
    local shadowMapMasked = {shadowMap,mask3D} - nn.CMulTable()
    local shadowMapModified = {shadowMapMasked,shadowMapBlurredMasked} - nn.CAddTable()
    local shadowMapModifiedReciprocal = shadowMapModified - nn.Power(-1)
    local outputModified = {input_deprocess,shadowMapModifiedReciprocal} - nn.CMulTable()

    local output_ = output - nn.MulConstant(2) - nn.AddConstant(-1)
    local outputModified_ = outputModified - nn.MulConstant(2) - nn.AddConstant(-1)
    local outputmasked_ = outputmasked - nn.MulConstant(2) - nn.AddConstant(-1)
    local shadowMapModified_ = shadowMapModified - nn.MulConstant(2) - nn.AddConstant(-1)
    local shadowMap_ = shadowMap - nn.MulConstant(2) - nn.AddConstant(-1)
    local mask3D_ = mask3D - nn.MulConstant(2) - nn.AddConstant(-1)
    local mask3D_soft_ = mask3D_soft

    netG = nn.gModule({input},{output_, outputmasked_, shadowMap_, outputModified_,shadowMapModified_,mask3D_,mask3D_soft_})
    return netG
end

function defineG_unet_128(input_nc, output_nc, ngf)
    -- Two layer less than the default unet to handle 128x128 input
    local netG = nil
    -- input is (nc) x 128 x 128
    local e1 = - nn.SpatialConvolution(input_nc, ngf, 4, 4, 2, 2, 1, 1)
    -- input is (ngf) x 64 x 64
    local e2 = e1 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)
    -- input is (ngf * 2) x 32 x 32
    local e3 = e2 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 2, ngf * 4, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 4)
    -- input is (ngf * 4) x 16 x 16
    local e4 = e3 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 4, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 8 x 8
    local e5 = e4 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 4 x 4
    local e6 = e5 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 2 x 2
    local e7 = e6 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8)
    -- input is (ngf * 8) x 1 x 1
    
    local d1_ = e7 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
    -- input is (ngf * 8) x 2 x 2
    local d1 = {d1_,e6} - nn.JoinTable(2)
    local d2_ = d1 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
    -- input is (ngf * 8) x 4 x 4
    local d2 = {d2_,e5} - nn.JoinTable(2)
    local d3_ = d2 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 8, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 8) - nn.Dropout(0.5)
    -- input is (ngf * 8) x 8 x 8
    local d3 = {d3_,e4} - nn.JoinTable(2)
    local d4_ = d3 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 8 * 2, ngf * 4, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 4)
    -- input is (ngf * 8) x 16 x 16
    local d4 = {d4_,e3} - nn.JoinTable(2)
    local d5_ = d4 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 4 * 2, ngf * 2, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * 2)
    -- input is (ngf * 4) x 32 x 32
    local d5 = {d5_,e2} - nn.JoinTable(2)
    local d6_ = d5 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2 * 2, ngf, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf)
    -- input is (ngf * 2) x 64 x 64
    local d6 = {d6_,e1} - nn.JoinTable(2)
    local d7 = d6 - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * 2, output_nc, 4, 4, 2, 2, 1, 1)
    -- input is (ngf) x128 x 128
    
    local o1 = d7 - nn.Tanh()
    
    netG = nn.gModule({e1},{o1})
    
    --graph.dot(netG.fg,'netG')
    
    return netG
end

function defineD_basic(input_nc, output_nc, ndf)
    n_layers = 3
    return defineD_n_layers(input_nc, output_nc, ndf, n_layers)
end

-- rf=1
function defineD_pixelGAN(input_nc, output_nc, ndf)
    local netD = nn.Sequential()
    
    -- input is (nc) x 256 x 256
    netD:add(nn.SpatialConvolution(input_nc+output_nc, ndf, 1, 1, 1, 1, 0, 0))
    netD:add(nn.LeakyReLU(0.2, true))
    -- state size: (ndf) x 256 x 256
    netD:add(nn.SpatialConvolution(ndf, ndf * 2, 1, 1, 1, 1, 0, 0))
    netD:add(nn.SpatialBatchNormalization(ndf * 2)):add(nn.LeakyReLU(0.2, true))
    -- state size: (ndf*2) x 256 x 256
    netD:add(nn.SpatialConvolution(ndf * 2, 1, 1, 1, 1, 1, 0, 0))
    -- state size: 1 x 256 x 256
    netD:add(nn.Sigmoid())
    -- state size: 1 x 30 x 30
        
    return netD
end

-- if n=0, then use pixelGAN (rf=1)
-- else rf is 16 if n=1
--            34 if n=2
--            70 if n=3
--            142 if n=4
--            286 if n=5
--            574 if n=6
function defineD_n_layers(input_nc, output_nc, ndf, n_layers)
    if n_layers==0 then
        return defineD_pixelGAN(input_nc, output_nc, ndf)
    else
    
        local netD = nn.Sequential()
        
        -- input is (nc) x 256 x 256
        netD:add(nn.SpatialConvolution(input_nc+output_nc, ndf, 4, 4, 2, 2, 1, 1))
        netD:add(nn.LeakyReLU(0.2, true))
        
        local nf_mult = 1
        local nf_mult_prev = 1
        for n = 1, n_layers-1 do 
            nf_mult_prev = nf_mult
            nf_mult = math.min(2^n,8)
            netD:add(nn.SpatialConvolution(ndf * nf_mult_prev, ndf * nf_mult, 4, 4, 2, 2, 1, 1))
            netD:add(nn.SpatialBatchNormalization(ndf * nf_mult)):add(nn.LeakyReLU(0.2, true))
        end
        
        -- state size: (ndf*M) x N x N
        nf_mult_prev = nf_mult
        nf_mult = math.min(2^n_layers,8)
        netD:add(nn.SpatialConvolution(ndf * nf_mult_prev, ndf * nf_mult, 4, 4, 1, 1, 1, 1))
        netD:add(nn.SpatialBatchNormalization(ndf * nf_mult)):add(nn.LeakyReLU(0.2, true))
        -- state size: (ndf*M*2) x (N-1) x (N-1)
        netD:add(nn.SpatialConvolution(ndf * nf_mult, 1, 4, 4, 1, 1, 1, 1))
        -- state size: 1 x (N-2) x (N-2)
        
        netD:add(nn.Sigmoid())
        -- state size: 1 x (N-2) x (N-2)
        
        return netD
    end
end
