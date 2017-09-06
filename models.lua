require 'nngraph'
require 'gvnn'
--require 'spy'

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
    local e8 = e7 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) -- nn.SpatialBatchNormalization(ngf * 8)
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
    local e8 = e7 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) -- nn.SpatialBatchNormalization(ngf * 8)
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

function defineG_unet_dewarp(input_nc, output_nc, ngf, height, width)
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
    local e8 = e7 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) -- nn.SpatialBatchNormalization(ngf * 8)
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
    local spfullconv_zero = nn.SpatialFullConvolution(ngf * 2, 2, 4, 4, 2, 2, 1, 1)
    spfullconv_zero.weight:fill(0.0)
    spfullconv_zero.bias:fill(0.0)
    local d8 = d7 - nn.ReLU(true) - spfullconv_zero -- nn.SpatialFullConvolution(ngf * 2, 2, 4, 4, 2, 2, 1, 1)
    -- input is (nc) x 256 x 256

    --local o1 = d8 - nn.Tanh()
    local offsetSobelX_2 = d8 - nn.SobelXConv(2,2) - nn.Abs()--nn.Square()
    local offsetSobelY_2 = d8 - nn.SobelYConv(2,2) - nn.Abs()--nn.Square()
    local offsetSobel = {offsetSobelX_2,offsetSobelY_2} - nn.CAddTable() nn.MulConstant(0.5) -- nn.Sqrt() - nn.MulConstant(0.71) - nn.Power(0.3)

    local d8_transposed = d8 - nn.Transpose({3,4},{2,4})
    local flowGrid = d8_transposed - nn.OpticalFlow2DBHWD(height,width)
    local inp_transposed = input - nn.Transpose({3,4},{2,4})
    local o1 = {inp_transposed,flowGrid} - nn.BilinearSamplerBHWD() - nn.Transpose({2,4},{3,4})
    netG = nn.gModule({input},{o1,offsetSobel})

    --graph.dot(netG.fg,'netG')

    return netG
end


do
    local RandomFlow2D, Parent = torch.class('nn.RandomFlow2D', 'nn.Module')

    function RandomFlow2D:__init(low, high)
       Parent.__init(self)
       self.low = low or 0
       self.high = high or 0.1
    end

    function RandomFlow2D:updateOutput(input)
       local shape = input:size()
       shape[shape:size()-2] = 2
       self.output:resize(shape):uniform(self.low, self.high)
       return self.output
    end
    
    function RandomFlow2D:updateGradInput(input, gradOutput)
       self.gradInput:resizeAs(input):zero()
       return self.gradInput
    end
end


function defineG_unet_flowPrediction(input_nc, output_nc, ngf, nLayers)
    local netG = nil
    local input = - nn.Identity()
    local input_flowOffsets = - nn.Identity()
    local feed_input = {input,input_flowOffsets} - nn.JoinTable(2)
  
    local encArr = {}
    local mulArr = {}
    encArr[1] = feed_input - nn.SpatialConvolution(input_nc+2, ngf, 4, 4, 2, 2, 1, 1)
    mulArr[1] = 1
    for i=2,nLayers do
	mulArr[i] = mulArr[i-1] * 2
	if mulArr[i] > 8 then mulArr[i] = 8 end
	local encLayer = encArr[i-1] - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * mulArr[i-1], ngf * mulArr[i], 4, 4, 2, 2, 1, 1)
	if i ~= nLayers then
	    encArr[i] = encLayer - nn.SpatialBatchNormalization(ngf * mulArr[i])
	else
	    encArr[i] = encLayer
	end
    end

    local decArr_ = {}
    decArr_[1] = encArr[nLayers] - nn.SpatialFullConvolution(ngf * mulArr[nLayers], ngf * mulArr[nLayers-1], 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * mulArr[nLayers-1]) - nn.Dropout(0.5) 
    for i=2,nLayers-1 do
	local dLayerJoin = {decArr_[i-1],encArr[nLayers-i+1]} - nn.JoinTable(2)
        local mulFrom = mulArr[nLayers-i+1]
	local mulTo = mulArr[nLayers-i]
	local dL = dLayerJoin - nn.ReLU(true) - nn.SpatialFullConvolution(ngf * mulFrom * 2, ngf * mulTo, 4, 4, 2, 2, 1, 1) - nn.SpatialBatchNormalization(ngf * mulTo)
	if i <= 3 then 
	    decArr_[i] = dL - nn.Dropout(0.5)
	else
	    decArr_[i] = dL
	end
    end

    local dLayerJoin = {decArr_[nLayers-1],encArr[1]} - nn.JoinTable(2)
    local spfullconv_zero = nn.SpatialFullConvolution(ngf * 2, 2, 4, 4, 2, 2, 1, 1)
    spfullconv_zero.weight:fill(0.0)
    spfullconv_zero.bias:fill(0.0)    
    decArr_[nLayers] = dLayerJoin - nn.ReLU(true) - spfullconv_zero
    
    local flowOffsets = {decArr_[nLayers],input_flowOffsets} - nn.CAddTable()

--[[
    -- input is (nc) x 256 x 256
    local e1 = feed_input - nn.SpatialConvolution(input_nc+2, ngf, 4, 4, 2, 2, 1, 1)
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
    local e8 = e7 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) -- nn.SpatialBatchNormalization(ngf * 8)
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
    local spfullconv_zero = nn.SpatialFullConvolution(ngf * 2, 2, 4, 4, 2, 2, 1, 1)
    spfullconv_zero.weight:fill(0.0)
    spfullconv_zero.bias:fill(0.0)
    local d8 = d7 - nn.ReLU(true) - spfullconv_zero -- nn.SpatialFullConvolution(ngf * 2, 2, 4, 4, 2, 2, 1, 1)

    local flowOffsets = {d8,input_flowOffsets} - nn.CAddTable()
]]--
    netG = nn.gModule({input,input_flowOffsets},{flowOffsets})
    return netG
end


function warp_by_flow(height, width)
    local netG = nil
    local input = - nn.Identity()
    local input_flowOffsets = - nn.Identity()

    local offsets_transposed = input_flowOffsets - nn.Transpose({3,4},{2,4})
    local flowGrid = offsets_transposed - nn.OpticalFlow2DBHWD(height,width)
    local inp_transposed = input - nn.Transpose({3,4},{2,4})
    local output = {inp_transposed,flowGrid} - nn.BilinearSamplerBHWD() - nn.Transpose({2,4},{3,4})

    netG = nn.gModule({input,input_flowOffsets},{output})
    return netG
end



function downsample_by2()
    local netG = nil
    local input = - nn.Identity()
    local output = input - nn.SpatialAveragePooling(2,2,2,2)
    netG = nn.gModule({input},{output})
    return netG
end


function dewarp_multiscale(input_nc, output_nc, ngf, height, width, nLevels)
    local netG = nil
    nLevels = nLevels or 1
    local input = - nn.Identity()
  
    local inputArr = {}
    inputArr[1] = input
    local scale = 1
    for i=2,nLevels do
	scale = scale * 2
	inputArr[i] = inputArr[i-1] - nn.SpatialAveragePooling(2,2,2,2)
    end

    local zeroOffsets = inputArr[nLevels] - nn.RandomFlow2D(0,0)
    local flowArr = {}
    local flowPredictorArr = {}
    flowPredictorArr[1] = defineG_unet_flowPrediction(input_nc, output_nc, ngf, 8 - nLevels + 1)
    flowArr[1] = {inputArr[nLevels],zeroOffsets} - flowPredictorArr[1]

    for i=2,nLevels do
	scale = scale / 2
	local initOffsets = flowArr[i-1] - nn.SpatialUpSamplingBilinear(2)
	--flowPredictorArr[i] = flowPredictorArr[1]:clone('weight','bias','gradWeight','gradBias')
        flowPredictorArr[i] = defineG_unet_flowPrediction(input_nc, output_nc, ngf, 8 - nLevels + i)
	local warping = warp_by_flow(height/scale, width/scale)
	--warping:evaluate()
	local waped_input = {inputArr[nLevels-i+1],initOffsets} - warping
        flowArr[i] = {waped_input,initOffsets} - flowPredictorArr[i]
    end
 
    local offsets = flowArr[nLevels]
    local warping = warp_by_flow(height,width)
    --warping:evaluate()
    local output = {input,offsets} - warping

    local offsetSobelX_2 = offsets - nn.SobelXConv(2,2) - nn.Abs()
    local offsetSobelY_2 = offsets - nn.SobelYConv(2,2) - nn.Abs()
    local offsetSobel = {offsetSobelX_2,offsetSobelY_2} - nn.CAddTable() nn.MulConstant(0.5)
 
    netG = nn.gModule({input},{output,offsetSobel,offsets})
    return netG

end

function define_loss(height, width)
    local net = nil
    local input_ = - nn.Identity()
    local GT_ = - nn.Identity()
    local flowBy2 = - nn.Identity()
    local flowBy4 = - nn.Identity()

    local input = input_ - nn.AddConstant(1) - nn.MulConstant(0.5)
    local GT = GT_ - nn.AddConstant(1) - nn.MulConstant(0.5)
    local inputBy2 = input - nn.SpatialAveragePooling(2,2,2,2)
    local inputBy4 = inputBy2 - nn.SpatialAveragePooling(2,2,2,2)
    local GTby2 = GT - nn.SpatialAveragePooling(2,2,2,2)
    local GTby4 = GTby2 - nn.SpatialAveragePooling(2,2,2,2)

    local dewarpBy2 = {inputBy2,flowBy2} - warp_by_flow(height/2, width/2)
    local dewarpBy4 = {inputBy4,flowBy4} - warp_by_flow(height/4, width/4)

    local errBy2 = {GTby2,dewarpBy2} - nn.CSubTable() - nn.Abs()
    local errBy4 = {GTby4,dewarpBy4} - nn.CSubTable() - nn.Abs()
    net = nn.gModule({input_,GT_,flowBy2,flowBy4},{errBy2,errBy4})
    return net
end

function dewarp_multiStep(input_nc, output_nc, ngf, height, width, nLevels)
    local netG = nil
    nLevels = nLevels or 1
    local input = - nn.Identity()

    local zeroOffsets = input - nn.RandomFlow2D(0,0)
    local flowArr = {}
    local flowPredictorArr = {}
    flowPredictorArr[1] = defineG_unet_flowPrediction(input_nc, output_nc, ngf, 8)
    flowArr[1] = {input,zeroOffsets} - flowPredictorArr[1]

    for i=2,nLevels do
	local initOffsets = flowArr[i-1]
	flowPredictorArr[i] = flowPredictorArr[1]:clone('weight','bias','gradWeight','gradBias')
	local warping = warp_by_flow(height, width)
	local waped_input = {input,initOffsets} - warping
	flowArr[i] = {waped_input,initOffsets} - flowPredictorArr[i]
    end

    local offsets = flowArr[nLevels]
    local warping = warp_by_flow(height,width)
    --warping:evaluate()
    local output = {input,offsets} - warping

    local offsetSobelX_2 = offsets - nn.SobelXConv(2,2) - nn.Abs()
    local offsetSobelY_2 = offsets - nn.SobelYConv(2,2) - nn.Abs()
    local offsetSobel = {offsetSobelX_2,offsetSobelY_2} - nn.CAddTable() nn.MulConstant(0.5)

    netG = nn.gModule({input},{output,offsetSobel})
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
    local e7 = e6 - nn.LeakyReLU(0.2, true) - nn.SpatialConvolution(ngf * 8, ngf * 8, 4, 4, 2, 2, 1, 1) -- nn.SpatialBatchNormalization(ngf * 8)
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
